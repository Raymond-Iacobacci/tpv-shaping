import os
import random
from pathlib import Path

import ff
import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import argparse

# --------------------------------------------------
# Neural network generator: maps latent vector to grating pattern
# --------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, n_elements):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(64, n_elements)
        )

    def forward(self, z):
        # Sigmoid to ensure outputs in [0,1]
        return torch.sigmoid(self.model(z))

# --------------------------------------------------
# Physics-based gradient + full-field sampling
# --------------------------------------------------
h=1
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float,
                       plot_fields: bool = False):
    p = 350
    n_grating_elements = grating.shape[-1]
    x_density = 10
    n_x_pts = x_density * n_grating_elements
    depth = 0.7
    vac_depth = 1.8
    z_space = np.linspace(0, vac_depth + depth + 1, 40)
    # measurement volume for gradient: within grating layer
    z_meas = z_space[(z_space >= vac_depth) & (z_space <= vac_depth + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)
    z_buf = 0.

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2 * k, 2) / (2 * k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()

    x_space = make_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux = torch.zeros((2, n_x_pts))
    dflux1 = torch.zeros((2, n_x_pts))
    power = []
    N = 3
    for i_wl, wl in enumerate(wavelengths[p:p + 1]):
        # forward solve
        S = S4.New(Lattice=L, NumBasis=N)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + p + 130]**2)
        # S.SetMaterial(Name='W',   Epsilon=(ff.w_n[i_wl+p+130]**2-1)*grating[0].item()+1)

        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl + p + 130]**2 - 1) * grating[0].item() + 1)
        # S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl+p+130]**2-1)+1)

        S.AddLayer(Name='VacuumAbove', Thickness=vac_depth, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='Vac')
        S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (L/2.1, L/2.), Angle = 0)
        S.AddLayer(Name='Ab', Thickness=1.0, Material='Vac')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave((0, 0),
                                 sAmplitude=np.cos(ang_pol * np.pi/180),
                                 pAmplitude=np.sin(ang_pol * np.pi/180),
                                 Order=0)
        forw, back = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(np.abs(back))

        S_adj = S.Clone()
        S_adj_homo = S.Clone()
        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buf)
        # Inhomogeneous adjoint simulation
        excitations = []
        # for i, raw_amp in enumerate(back_amp[:int(len(back_amp)/2)]):
        # print(S.GetBasisSet())
        # sys.exit(1)
        num_nodes = len(S.GetBasisSet())
        # print(back_amp)
        for i, raw_amp in enumerate(back_amp[1:2]):
        # for idx in (1,):
            k0 = 2 * np.pi / (wl.item())
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            excitations.append((h, b'y', corr_amp))
        S_adj.SetExcitationExterior(Excitations=tuple(excitations)) # This is the problem -- it's not accurate with higher harmonics and disappears at the 0-harmonic
        # print(excitations)
        k0 = 2 * np.pi / (wl.item())
        phase_correction = np.exp(-1j * k0 * z_buf)   # one-way trip
        adj_amp = phase_correction * np.conj(back_amp[0])
        # S_adj.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180) * adj_amp,pAmplitude=np.sin(ang_pol*np.pi/180) * adj_amp,Order=0)
        # print(adj_amp)


        # plot if requested
        if plot_fields:

            # sample full forward volume
            fwd_vol = np.zeros((z_space.size, n_x_pts), complex)
            for iz, z in enumerate(z_space):
                for ix, x in enumerate(x_space):
                    fwd_vol[iz, ix] = S.GetFields(x, 0, z)[0][1]

            adj_vol = np.zeros((z_space.size, n_x_pts), complex)
            for iz, z in enumerate(z_space):
                for ix, x in enumerate(x_space):
                    adj_vol[iz, ix] = S_adj.GetFields(x, 0, z)[0][1]

            # compute product of forward and adjoint fields
            prod_vol = fwd_vol * (adj_vol)

            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            titles = [
                f'Forward Real E-field {grating[0]}', f'Forward Imag E-field {vac_depth}',
                'Adjoint Real E-field', 'Adjoint Imag E-field',
                'Product Real (E·E_adj)', 'Product Imag (E·E_adj)'
            ]
            data_list = [
                np.real(fwd_vol), np.imag(fwd_vol),
                np.real(adj_vol), np.imag(adj_vol),
                np.real(prod_vol), np.imag(prod_vol)
            ]

            for ax, title, data in zip(axs.ravel(), titles, data_list):
                im = ax.imshow(
                    data,
                    extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                    aspect='auto'
                )
                ax.set_title(f'{title} {grating[0]}' if 'Forward' in title else title)
                ax.set_xlabel('x (um)')
                if 'Real' in title:
                    ax.set_ylabel('z (um)')
                # layer boundaries
                ax.axhline(vac_depth, color='white', linestyle='--', linewidth=1)
                ax.axhline(vac_depth + depth, color='white', linestyle='--', linewidth=1)
                fig.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.show()

        vol_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                vol_meas[iz, 0, ix] = S.GetFields(x, 0, z)[0]
                adj_meas[iz, 0, ix] = S_adj.GetFields(x, 0, z)[0]

        del S_adj

        rot = np.e ** (1j/2*np.pi) # Use this from homogeneous tests
        term = k0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(vol_meas),
                         torch.as_tensor(adj_meas)*rot)
        )
        dz = (depth) / len(z_meas)
        dflux[i_wl] = term.sum(dim=0).squeeze() * dz * L / n_x_pts
        # dflux[i_wl] = dflux[i_wl].reshape(-1, x_density).sum(dim=1)
        dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl + p + 130]**2 - 1) # HERE IT IS!


        # Homogeneous adjoint simulation
        k0 = 2 * np.pi / (wl.item())
        phase_correction = np.exp(-1j * k0 * z_buf)   # one-way trip
        adj_amp = phase_correction * np.conj(back_amp[0])
        S_adj_homo.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180) * adj_amp,pAmplitude=np.sin(ang_pol*np.pi/180) * adj_amp,Order=0)

        homo_adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                homo_adj_meas[iz, 0, ix] = S_adj_homo.GetFields(x, 0, z)[0]
        
        term1 = k0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(vol_meas),
                         torch.as_tensor(homo_adj_meas)*rot)
        )
        dflux1[i_wl] = term1.sum(dim=0).squeeze() * dz * L / n_x_pts
        dflux1[i_wl] = dflux1[i_wl] * (ff.aln_n[i_wl + p + 130]**2 - 1) # HERE IT IS!

        # dflux = dflux + dflux1


    dfom = dflux[0]
    diff = float(power[0])
    # print(dflux1[0][2:8].shape)
    # sys.exit(1)
    return (2*torch.sum(dfom[3:7]), 2*torch.sum(dflux1[0][3:7])), diff

# --------------------------------------------------
# Main: scan & plot
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scan gradient and optionally plot full fields")
    parser.add_argument('--plot-fields', action='store_true',
                        help='Plot full-field forward and adjoint E-fields')
    args = parser.parse_args()

    L = 1.
    ang_pol = 0.0
    step = 0.01
    i_vals = np.arange(0.0, 1.0 + step, step)
    grad_vals = []
    grad_vals1 = []
    P=[]
    for val in tqdm(i_vals, desc="Scanning gradient"):
        g = torch.tensor([val], dtype=torch.float32)
        (dfom, dfom1), P1 = gradient_per_image(g, L, ang_pol, plot_fields=args.plot_fields)
        grad_vals.append(dfom.item())
        grad_vals1.append(dfom1.item())
        P.append(P1)
    plt.plot(P)
    plt.show()
    correct_slopes = np.load('slope.npy')
    plt.figure(figsize=(6, 4))
    plt.plot(i_vals, grad_vals, lw=2)
    plt.xlabel('Grating amplitude')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title(f'Gradient of FOM vs. Grating Amplitude {h}')
    plt.grid(True)
    plt.show()

    correct_slopes = np.load('slope.npy')
    plt.figure(figsize=(6, 4))
    plt.plot(i_vals, grad_vals1, lw=2)
    plt.xlabel(f'Imaginary Grating amplitude {h}')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title('Gradient of FOM vs. Grating Amplitude')
    plt.grid(True)
    plt.show()

    thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    scalings = np.linspace(0.3, 1.5, 101)
    best_mse = float('inf')
    best_theta = 0.0
    best_scale = 1.0
    best_weighted = None

    for theta in thetas:
        unit_w = np.cos(theta) + 1j * np.sin(theta)
        w = unit_w
        slopes_variant = w.real * np.array(grad_vals) + w.imag * np.array(grad_vals1)
        mse = np.mean(np.abs(slopes_variant - correct_slopes)**2)
        if mse < best_mse:
            best_mse = mse
            best_theta = theta
            best_weighted = slopes_variant

    print(f'Optimal scale: {best_scale:.3f}, theta: {best_theta:.3f} rad')
    print(f'Optimal complex weight: (cos+isin) -> {best_weighted[0].real + 1j*best_weighted[0].imag}, MSE: {best_mse}')
    # Plot best weighted vs correct
    plt.figure(figsize=(6, 4))
    plt.plot(i_vals, correct_slopes, label='Correct slopes', lw=2)
    plt.plot(i_vals, grad_vals, label=f'Weighted theta={best_theta:.3f})', lw=2)
    plt.xlabel('Grating amplitude')
    plt.ylabel('Slope')
    plt.title('Best weighted slopes vs. correct slopes')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(np.max(np.abs(correct_slopes[1:-1] - grad_vals[1:-1])))
    print(correct_slopes[10] , grad_vals[10])
    print(correct_slopes[50] , grad_vals[50])

if __name__ == '__main__':
    main()
