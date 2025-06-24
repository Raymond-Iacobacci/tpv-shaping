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
    p = 1480
    n_grating_elements = grating.shape[-1]
    x_density = 10
    n_x_pts = x_density * n_grating_elements
    depth = 0.7
    z_space = np.linspace(0, 0.5 + depth + 1, 200)
    # measurement volume for gradient: within grating layer
    z_meas = z_space[(z_space >= 0.5) & (z_space <= 0.5 + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2 * k, 2) / (2 * k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()

    x_space = make_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux = torch.zeros((2, n_x_pts))
    power = []

    for i_wl, wl in enumerate(wavelengths[p:p + 2]):
        # forward solve
        S = S4.New(Lattice=L, NumBasis=20)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + p + 130]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl + p + 130]**2 - 1) * grating[0].item() + 1)
        S.AddLayer(Name='VacuumAbove', Thickness=0.5, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='AlN')
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave((0, 0),
                                 sAmplitude=np.cos(ang_pol * np.pi/180),
                                 pAmplitude=np.sin(ang_pol * np.pi/180),
                                 Order=0)
        forw, back = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(1 - np.abs(back))

        # sample full forward volume
        fwd_vol = np.zeros((z_space.size, n_x_pts), complex)
        for iz, z in enumerate(z_space):
            for ix, x in enumerate(x_space):
                fwd_vol[iz, ix] = S.GetFields(x, 0, z)[0][1]

        # adjoint solve
        S_adj = S.Clone()
        # build adjoint excitation at buffer
        adj_buf = np.array([S.GetFields(x, 0, 0.0)[0][1]-1 for x in x_space])
        mag = np.abs(adj_buf)**2
        phase = np.angle(adj_buf)
        step = mag * np.exp(-1j * phase)
        fourier_adj = ff.create_step_excitation(
            basis=S_adj.GetBasisSet(), step_values=step,
            num_harmonics=20, x_shift=0, initial_phase=0,
            amplitude=1.0, period=L, plot_fourier=False
        )
        S_adj.SetExcitationExterior(Excitations=tuple(fourier_adj))

        # sample full adjoint volume
        adj_vol = np.zeros((z_space.size, n_x_pts), complex)
        for iz, z in enumerate(z_space):
            for ix, x in enumerate(x_space):
                adj_vol[iz, ix] = S_adj.GetFields(x, 0, z)[0][1]

        # plot if requested
        if plot_fields:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            im0 = axs[0,0].imshow(np.real(fwd_vol),
                                   extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                                   aspect='auto')
            axs[0,0].set_title('Forward Real E-field')
            axs[0,0].set_xlabel('x (um)')
            axs[0,0].set_ylabel('z (um)')
            fig.colorbar(im0, ax=axs[0,0])

            im1 = axs[0,1].imshow(np.imag(fwd_vol),
                                   extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                                   aspect='auto')
            axs[0,1].set_title('Forward Imag E-field')
            axs[0,1].set_xlabel('x (um)')
            fig.colorbar(im1, ax=axs[0,1])

            im2 = axs[1,0].imshow(np.real(adj_vol),
                                   extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                                   aspect='auto')
            axs[1,0].set_title('Adjoint Real E-field')
            axs[1,0].set_xlabel('x (um)')
            axs[1,0].set_ylabel('z (um)')
            fig.colorbar(im2, ax=axs[1,0])

            im3 = axs[1,1].imshow(np.imag(adj_vol),
                                   extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                                   aspect='auto')
            axs[1,1].set_title('Adjoint Imag E-field')
            axs[1,1].set_xlabel('x (um)')
            fig.colorbar(im3, ax=axs[1,1])

            plt.tight_layout()
            plt.show()

        # gradient computation (unchanged)
        vol_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                vol_meas[iz, 0, ix] = S.GetFields(x, 0, z)[0]
                adj_meas[iz, 0, ix] = S_adj.GetFields(x, 0, z)[0]

        del S, S_adj

        k0 = 2 * np.pi / (wl.item() * 1e-6)
        term = 2 * k0**2 * ff.e_0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(vol_meas),
                         torch.as_tensor(adj_meas).conj())
        )
        dz = (z_meas[-1] - z_meas[0]) * 1e-6
        dflux[i_wl] = term.sum(dim=0).squeeze() * dz
        dflux[i_wl] = dflux[i_wl].reshape(-1, x_density).sum(dim=1)
        dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl + p + 130]**2 - 1) / np.real(forw)

    dfom = dflux[0]
    diff = float(power[0])
    return dfom, diff

# --------------------------------------------------
# Main: scan & plot
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scan gradient and optionally plot full fields")
    parser.add_argument('--plot-fields', action='store_true',
                        help='Plot full-field forward and adjoint E-fields')
    args = parser.parse_args()

    L = 1.1
    ang_pol = 0.0
    step = 0.01
    i_vals = np.arange(0.0, 1.0 + step, step)
    grad_vals = []

    for val in tqdm(i_vals, desc="Scanning gradient"):
        g = torch.tensor([val], dtype=torch.float32)
        dfom, _ = gradient_per_image(g, L, ang_pol, plot_fields=args.plot_fields)
        grad_vals.append(dfom[0].item())

    plt.figure(figsize=(6, 4))
    plt.plot(i_vals, grad_vals, lw=2)
    plt.xlabel('Grating amplitude')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title('Gradient of FOM vs. Grating Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
