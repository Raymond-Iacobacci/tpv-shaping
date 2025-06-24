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
            nn.Softmax(dim=-1),
            # nn.ReLU(inplace=True),
            nn.Linear(64, n_elements)
        )

    def forward(self, z):
        # Sigmoid to ensure outputs in [0,1]
        return torch.sigmoid(self.model(z))

# --------------------------------------------------
# Physics-based gradient: computes dfom_deps and FOM diff
# --------------------------------------------------
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float):
    p = 1480
    n_grating_elements = grating.shape[-1]
    x_density = 5
    n_x_pts = x_density * n_grating_elements
    depth = 0.7
    z_buff = 0.15
    z_space = np.linspace(0, 0.5 + depth + 1, 10)
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

    for i_wl, wl in enumerate(wavelengths[p:p+2]):
        S = S4.New(Lattice=L, NumBasis=20)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl+130+p]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=ff.aln_n[i_wl+130+p]**2)

        S.AddLayer(Name='VacuumAbove', Thickness=0.5, Material='Vac')
        S.AddLayer(Name='Grating',      Thickness=depth, Material='Vac')
        for ns in range(n_grating_elements):
            eps = grating[ns].item() * (ff.aln_n[i_wl+130+p]**2 - 1) + 1
            S.SetMaterial(Name=f'sq{ns+1}', Epsilon=eps)
            ctr = (((ns+1)/n_grating_elements - 1/(2*n_grating_elements))*L, 0.5)
            hw = ((1/(2*n_grating_elements))*L, 0.5)
            S.SetRegionRectangle(
                Layer='Grating', Material=f'sq{ns+1}',
                Center=ctr, Halfwidths=hw, Angle=0
            )
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)

        S_adj = S.Clone()
        S.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180), pAmplitude=np.sin(ang_pol*np.pi/180), Order=0)
        (_, back) = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(1 - np.abs(back))

        adj_flds = np.zeros((1,1,n_x_pts), complex)
        for ix, x in enumerate(x_meas): adj_flds[0,0,ix] = S.GetFields(x,0,z_buff)[0][1]
        vol = np.zeros((z_meas.size,1,n_x_pts,3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_meas): vol[iz,0,ix] = S.GetFields(x,0,z)[0]
        del S

        mag   = np.abs(adj_flds[0,0,:]**2)
        phase = np.angle(adj_flds[0,0,:])
        step  = mag * np.exp(1j * phase)
        fourier_adj = ff.create_step_excitation(
            basis=S_adj.GetBasisSet(), step_values=step,
            num_harmonics=30, x_shift=0, initial_phase=0,
            amplitude=1.0, period=L, plot_fourier=False
        )
        S_adj.SetExcitationExterior(Excitations=tuple(fourier_adj))

        adj_vol = np.zeros((z_meas.size,1,n_x_pts,3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_meas): adj_vol[iz,0,ix] = S_adj.GetFields(x,0,z)[0]

        term = 2 * (wl/1e6)**-2 * ff.e_0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk', torch.as_tensor(vol), torch.as_tensor(adj_vol).conj())
        )
        dflux[i_wl] = term.mean(dim=0).squeeze() * (ff.aln_n[i_wl+130+p]**2 - 1)
        del S_adj

    dfom = dflux[0] - dflux[1]
    dfom = dfom.reshape(-1, x_density).sum(dim=1)
    diff = float(power[0] - power[1])
    return dfom, diff

# --------------------------------------------------
# Main optimization: train generator network & live-plot diffs
# --------------------------------------------------
def main():
    n_gratings = 1
    latent_dim = 100
    n_elements = 10
    lr = 1e-2
    n_epochs = 1000
    L = 1.1
    ang_pol = 0.0

    latents = torch.randn(n_gratings, latent_dim)
    net = Generator(latent_dim, n_elements)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    diff_hist = [[] for _ in range(n_gratings)]
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    lines = [ax.plot([], [], label=f'Seed {i}')[0] for i in range(n_gratings)]
    ax.set_xlim(1, n_epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diff')
    ax.set_title('Live Diff Over Time')
    ax.legend()

    for epoch in tqdm(range(1, n_epochs+1), desc='Epochs'):
        optimizer.zero_grad()
        out = net(latents)  # (n_gratings, n_elements)

        grads = []
        diffs = []
        for i in range(n_gratings):
            dfom, diff = gradient_per_image(out[i], L, ang_pol)
            grads.append(dfom)
            diffs.append(diff)

        grad_mat = torch.stack(grads, dim=0)

        out.backward(gradient=grad_mat)
        optimizer.step()

        for i, d in enumerate(diffs):
            diff_hist[i].append(d)
            xs = list(range(1, len(diff_hist[i]) + 1))
            lines[i].set_data(xs, diff_hist[i])
        all_d = [d for h in diff_hist for d in h]
        ax.set_ylim(min(all_d), max(all_d))

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{n_epochs}: diffs = {diffs}")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
