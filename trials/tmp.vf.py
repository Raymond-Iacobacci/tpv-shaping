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
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float,
                       plot_fields: bool = False):
    p = 350
    n_grating_elements = grating.shape[-1]
    x_density = 10
    n_x_pts = x_density * n_grating_elements
    depth = 0.7
    vac_depth = 1.8
    z_space = np.linspace(0, vac_depth + depth + 1, 70)
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
    power = []
    N = 1
    for i_wl, wl in enumerate(wavelengths[p:p + 1]):
        S = S4.New(Lattice=L, NumBasis=N)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + p + 130]**2)

        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl + p + 130]**2 - 1) * grating[0].item() + 1)

        S.AddLayer(Name='VacuumAbove', Thickness=vac_depth, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='Vac')
        S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (L/4, L/2), Angle = 0)
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave((0, 0),
                                 sAmplitude=np.cos(ang_pol * np.pi/180),
                                 pAmplitude=np.sin(ang_pol * np.pi/180),
                                 Order=0)
        forw, back = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(np.abs(back))

        fwd_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                fwd_meas[iz, 0, ix] = S.GetFields(x, 0, z)[0]

        S_adj_pos = S.Clone()
        S_adj_neg = S.Clone()
        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buf)

        if False:
            k0 = 2 * np.pi / (wl.item())
            phase_correction = np.exp(-1j * k0 * z_buf)
            adj_amp = phase_correction * np.conj(back_amp[0])
            S_adj = S.Clone()
            S_adj.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180) * adj_amp,pAmplitude=np.sin(ang_pol*np.pi/180) * adj_amp,Order=0)
            adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
            for iz, z in enumerate(z_meas):
                for ix, x in enumerate(x_space):
                    adj_meas[iz, 0, ix] = S_adj.GetFields(x, 0, z)[0]
            rot = np.e ** (1j/2*np.pi)
            term = k0 * torch.real(
                torch.einsum('ijkl,ijkl->ijk',
                            torch.as_tensor(fwd_meas),
                            torch.as_tensor(adj_meas)*rot) # This rotation factor is equivalent to taking the -imaginary value of the system
            )
            dz = (depth) / len(z_meas)
            dflux[i_wl] = term.sum(dim=0).squeeze() * dz * L / n_x_pts
            dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl + p + 130]**2 - 1)
            continue

        # S4 normalizes incident harmonics, so we must un-normalize the outputs before adding them back together
        print(back_amp)
        pos_excitations = []
        basis = S_adj_pos.GetBasisSet() # Removes the repeated calls
        pos_harmonics = [ i for i in range(len(basis)) if basis[i][0] > 0] # TODO: fix
        k0 = 2 * np.pi / wl.item()
        pos_norm = 0
        for i, raw_amp in enumerate(back_amp):
            if i not in pos_harmonics:
                continue
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            pos_excitations.append((basis[i][0]+1, b'y', corr_amp))
            pos_norm += np.abs(raw_amp) ** 2
        pos_norm = np.sqrt(pos_norm)
        S_adj_pos.SetExcitationExterior(tuple(pos_excitations))

        pos_adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                pos_adj_meas[iz, 0, ix] = S_adj_pos.GetFields(x, 0, z)[0]

        neg_excitations = []
        neg_harmonics = [ i for i in range(len(basis)) if basis[i][0] < 0]
        neg_norm = 0
        for i, raw_amp in enumerate(back_amp):
            if i not in neg_harmonics:
                continue
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            neg_excitations.append((-basis[i][0]+1, b'y', -corr_amp))
            neg_norm += np.abs(raw_amp) ** 2
        neg_norm = np.sqrt(neg_norm)
        S_adj_neg.SetExcitationExterior(tuple(neg_excitations))

        neg_adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                # neg_adj_meas[iz, 0, ix] = S_adj_neg.GetFields(x, 0, z)[0]
                neg_adj_meas[iz, 0, ix] = pos_adj_meas[iz, 0, -1-ix]

        rot = np.e ** (1j/2*np.pi) # Use this from homogeneous tests
        term = k0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(fwd_meas),
                         torch.as_tensor(pos_adj_meas * pos_norm + neg_adj_meas * neg_norm)*rot) # This rotation factor is equivalent to taking the -imaginary value of the system
        )
        dz = (depth) / len(z_meas)
        dflux[i_wl] = term.sum(dim=0).squeeze() * dz * L / n_x_pts
        dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl + p + 130]**2 - 1) # HERE IT IS!

        # print(pos_excitations)
        # print(neg_excitations)

        if plot_fields:

            fwd_vol = np.zeros((z_space.size, n_x_pts), complex)
            adj_vol = np.zeros((z_space.size, n_x_pts), complex)
            for iz, z in enumerate(z_space):
                for ix, x in enumerate(x_space):
                    fwd_vol[iz, ix] = S.GetFields(x, 0, z)[0][1]
                    adj_vol[iz, ix] = S_adj_pos.GetFields(x, 0, z)[0][1] + S_adj_pos.GetFields(-x, 0, z)[0][1]


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

        del S, S_adj_neg, S_adj_pos
    print(dflux)
    return (torch.sum(dflux[0][3:7])), power[0]

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
    P=[]
    for val in tqdm(i_vals, desc="Scanning gradient"):
        g = torch.tensor([val], dtype=torch.float32)
        dflux, P1 = gradient_per_image(g, L, ang_pol, plot_fields=args.plot_fields)
        grad_vals.append(dflux.item())
        P.append(P1)
    plt.plot(P)
    plt.show()
    correct_slopes = np.load('slope.npy')
    plt.figure(figsize=(6, 4))
    plt.plot(i_vals, grad_vals, label = 'Calculated slopes', lw=2)
    plt.plot(i_vals, correct_slopes, label = 'Correct slopes', lw = 2)
    plt.legend()
    plt.xlabel('Grating amplitude')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title('Gradient of FOM vs. Grating Amplitude')
    plt.grid(True)
    plt.show()

    print(np.max(np.abs(correct_slopes[1:-1] - grad_vals[1:-1])))
    print(correct_slopes[10] , grad_vals[10])
    print(correct_slopes[50] , grad_vals[50])

if __name__ == '__main__':
    main()
