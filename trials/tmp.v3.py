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
# Physics-based gradient: computes dfom_deps and FOM diff
# --------------------------------------------------
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float, rot):
    p = 1480
    # p = 1000
    n_grating_elements = grating.shape[-1]
    x_density = 1 # Not the problem
    n_x_pts = x_density * n_grating_elements
    depth = 0.7
    z_buff = 0.46
    # z_buff = 0.35
    z_space = np.linspace(0, 0.5 + depth + 1, 1000)
    z_meas = z_space[(z_space >= 0.5) & (z_space <= 0.5 + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2*k, 2) / (2*k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()
    x_meas = make_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux = torch.zeros((2, n_x_pts))
    power = []

    for i_wl, wl in enumerate(wavelengths[p:p+1]):
        S = S4.New(Lattice=L, NumBasis=20)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl+p+130]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl+p+130]**2-1)*grating[0].item()+1)

        S.AddLayer(Name='VacuumAbove', Thickness=0.9, Material='Vac')
        S.AddLayer(Name = 'Grating', Thickness = depth, Material = 'AlN')
        S.AddLayer(Name='VacuumBelow', Thickness=1, Material='Vac')
        # S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)

        S_adj = S.Clone()
        S.SetExcitationPlanewave((0,0),
            sAmplitude=np.cos(ang_pol*np.pi/180),
            pAmplitude=np.sin(ang_pol*np.pi/180),
            Order=0)
        (_, back) = S.GetPowerFlux('VacuumAbove', zOffset=0)
        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buff)
        # power.append(np.abs(back))
        '''
        bwd_reals, bwd_imags = [], []
        for zp in np.linspace(0, 0.5, 50):
            (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=.5-zp)
            bwd_reals.append(np.real(back_amp))
            bwd_imags.append(np.imag(back_amp))
        bwd_reals, bwd_imags = np.array(bwd_reals), np.array(bwd_imags)
        fig, (ax1, ax2,ax3) = plt.subplots(1,3)
        ax1.plot(bwd_reals)
        ax2.plot(bwd_imags)
        ax3.plot(bwd_reals**2 + bwd_imags**2)
        print(back)
        plt.show()
        '''

        # build adjoint fields…
        adj_flds = np.zeros((1,1,n_x_pts), complex)
        for ix, x in enumerate(x_meas):
            adj_flds[0,0,ix] = S.GetFields(x,0,z_buff)[0][1]
        vol = np.zeros((z_meas.size,1,n_x_pts,3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_meas):
                vol[iz,0,ix] = S.GetFields(x,0,z)[0]
        del S
        
        step = np.conj(adj_flds[0][0]-1) # Complex derivative, takes away the influence of the forward (incoming) field
        # r = S.GetAmplitudes('VacuumAbove', 0)[0][1]
        # print(np.abs(adj_flds[0][0]-1)**2+power[-1])
        # print(step)
        # step = np.conj(back_amp[0]) # this only computes the central amplitude...right?
        fourier_adj = ff.create_step_excitation(
            basis=S_adj.GetBasisSet(), step_values=step,
            num_harmonics=20, x_shift=0, initial_phase=0,
            amplitude=1.0, period=L, plot_fourier=False
        )
        # S_adj.SetExcitationExterior(Excitations=tuple(fourier_adj))
        adj_rescale = np.conj(adj_flds[0][0]-1)
        # power.append(np.abs(adj_rescale[0])**2)
        power.append(np.abs(back_amp[0])**2)
        # print(f"Adjoint rescale:\n{adj_rescale[0]}\nMag: {np.abs(adj_rescale[0])**2}")
        S_adj.SetExcitationPlanewave((0,0),
                                     sAmplitude=np.cos(ang_pol*np.pi/180) * np.conj(back_amp[0]),
                                     pAmplitude=np.sin(ang_pol*np.pi/180) * np.conj(back_amp[0]),
                                     Order=0)
        # S_adj.SetExcitationPlanewave((0,0), sAmplitude=np.conj(r),
        #                      pAmplitude=0.0, Order=0)

        adj_vol = np.zeros((z_meas.size,1,n_x_pts,3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_meas):
                adj_vol[iz,0,ix] = S_adj.GetFields(x,0,z)[0]

        w = 2 * np.pi / (wl.item())
        term = torch.real(
            torch.einsum(
                'ijkl,ijkl->ijk',
                torch.as_tensor(vol),
                torch.as_tensor(adj_vol)
            )
        )
        # * ff.c / w / ff.e_0 
        # term *= -w ** 2 * ff.e_0/2
        term *= w
        dx = L / n_x_pts
        dz = (depth) / len(z_meas)
        # print(dx, dz, w, ff.e_0, ff.c)
        dflux[i_wl] = term.sum(dim=0).squeeze() * dz * dx
        dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl+p+130]**2 - 1) 
        # sys.exit(1)
        del S_adj

    # dfom is the gradient wrt your grating[0]
    # dfom = dflux[0].reshape(-1, x_density).mean(dim=1)
    dfom = dflux[0]
    diff = float(power[0])
    # print(dfom)
    # sys.exit(1)
    return dfom, diff

# --------------------------------------------------
# New: scan the gradient vs. grating amplitude and plot it
# --------------------------------------------------
def main():
    L = 1.
    ang_pol = 0.0
    step = 0.01

    i_vals = np.arange(0.0, 1.0+step, step)


    grad_vals = []
    power = []

    for val in tqdm(i_vals, desc="Scanning gradient"):
        # for a single-element grating
        g = torch.tensor([val], dtype=torch.float32)
        dfom, p = gradient_per_image(g, L, ang_pol, 0)
        power.append(p)
        # dfom is a 1-element tensor → extract it
        grad_vals.append(dfom[0].item())

    plt.plot(power)
    plt.show()
    # now plot exactly like your first script did
    plt.figure(figsize=(6,4))
    plt.plot(i_vals, grad_vals, lw=2)
    plt.xlabel('Grating amplitude')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title('Gradient of FOM vs. Grating Amplitude')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
