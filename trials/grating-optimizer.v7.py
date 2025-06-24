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
# Physics-based gradient: computes dfom_deps for a given grating
# --------------------------------------------------
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float):
    """
    Compute the figure-of-merit gradient (dfom_deps) for a single grating configuration,
    plus the FOM difference for live plotting.
    Returns:
      - dfom: gradient tensor shape (n_grating_elements,)
      - diff: scalar FOM difference between two wavelengths
    """
    # Configuration parameters
    p = 1480
    n_grating_elements = grating.shape[-1]
    n_x_measurement_pts = int(5 * n_grating_elements)
    depth = 0.7
    z_buff = 0.15
    z_space = np.linspace(0, 0.5 + depth + 1, 10)
    z_measurement_space = z_space[(z_space >= 0.5) & (z_space <= 0.5 + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)
    x_density = 5

    # Measurement grid
    def make_measurement_grid(L, n_cells, k):
        dx = L / n_cells
        fractions = np.arange(1, 2*k, 2) / (2*k)
        starts = np.arange(n_cells)[:, None]
        return ((starts + fractions) * dx).ravel()
    x_measurement_space = make_measurement_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux_deps = torch.zeros((2, n_x_measurement_pts))
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
            center = (((ns+1)/n_grating_elements - 1/(2*n_grating_elements))*L, 0.5)
            halfw = ((1/(2*n_grating_elements))*L, 0.5)
            S.SetRegionRectangle(
                Layer='Grating', Material=f'sq{ns+1}',
                Center=center, Halfwidths=halfw, Angle=0
            )
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)

        S_adj = S.Clone()

        S.SetExcitationPlanewave(
            IncidenceAngles=(0,0),
            sAmplitude=np.cos(ang_pol*np.pi/180),
            pAmplitude=np.sin(ang_pol*np.pi/180),
            Order=0
        )
        (_, back) = S.GetPowerFlux(Layer='VacuumAbove', zOffset=0)
        power.append(1-np.abs(back))

        # forward fields
        adj_fields = np.zeros((1,1,n_x_measurement_pts), dtype=complex)
        for ix, x in enumerate(x_measurement_space):
            adj_fields[0,0,ix] = S.GetFields(x,0,z_buff)[0][1]
        # volume fields
        grating_fields = np.zeros((z_measurement_space.size,1,x_measurement_space.size,3), complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                grating_fields[iz,0,ix] = S.GetFields(x,0,z)[0]
        del S

        adj_mag   = np.abs(adj_fields[0,0,:]**2)
        adj_phase = np.angle(adj_fields[0,0,:])
        adj_step  = adj_mag * np.exp(1j * adj_phase)
        fourier_adj = ff.create_step_excitation(
            basis=S_adj.GetBasisSet(), step_values=adj_step,
            num_harmonics=30, x_shift=0, initial_phase=0,
            amplitude=1.0, period=L, plot_fourier=False
        )
        S_adj.SetExcitationExterior(Excitations=tuple(fourier_adj))

        adj_grating_fields = np.zeros((z_measurement_space.size,1,n_x_measurement_pts,3), complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                adj_grating_fields[iz,0,ix] = S_adj.GetFields(x,0,z)[0]

        dflux = 2 * (wl/1e6)**-2 * ff.e_0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(grating_fields),
                         torch.as_tensor(adj_grating_fields).conj())
        )
        dflux_deps[i_wl] = dflux.mean(dim=0).squeeze() * (ff.aln_n[i_wl+130+p]**2 - 1)
        del S_adj

    dfom = dflux_deps[0] - dflux_deps[1]
    dfom = dfom.reshape(-1, x_density).sum(dim=1)
    diff = (float(power[0] - power[1]))**2
    return dfom, diff

# --------------------------------------------------
# Main optimization: live-plot diffs and update seeds
# --------------------------------------------------
def main():
    n_gratings = 1
    n_elements = 10
    lr = 1e-1
    n_epochs = 1000
    L = 1.1
    ang_pol = 0.0

    seeds = torch.nn.Parameter(torch.randn(n_gratings, n_elements))
    optimizer = torch.optim.Adam([seeds], lr=lr)

    diff_hist = [[] for _ in range(n_gratings)]
    stagnation_patience = 20
    stagnation_tol = 1e-5
    noise_scale = 0.3
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=f'Seed {i}')[0] for i in range(n_gratings)]
    ax.set_xlim(1, n_epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diff')
    ax.set_title('Live Diff Over Time')
    ax.legend()

    for _, epoch in enumerate(tqdm(range(1, n_epochs+1), desc="Epochs", leave = False)):
        optimizer.zero_grad()
        sig = torch.sigmoid(seeds)

        grads = []
        diffs = []
        for i in range(n_gratings):
            dfom, diff = gradient_per_image(sig[i], L, ang_pol)
            scaled_grad = 2 * diff * dfom
            scale = 2**np.abs(diff/(np.mean(diff_hist[-1]) if epoch > 1 else 1e-3))
            grads.append(scaled_grad*min(scale,10) * sig[i] * (1 - sig[i]))
            # grads.append(scaled_grad * sig[i] * ( 1 - sig[i]))
            diffs.append(diff)

        # Perform seed update
        seeds.grad = torch.stack(grads, dim=0)
        optimizer.step()

        # Record diffs before plotting
        for i, d in enumerate(diffs):
            diff_hist[i].append(d)
            # Inject noise if stagnated
            if len(diff_hist[i]) > stagnation_patience:
                recent = diff_hist[i][-stagnation_patience:]
                if max(recent) - min(recent) < stagnation_tol:
                    noise = torch.randn_like(seeds[i]) * noise_scale
                    with torch.no_grad():
                        seeds[i] += noise

        # Update live plot
        for i, line in enumerate(lines):
            xdata = list(range(1, len(diff_hist[i]) + 1))
            line.set_data(xdata, diff_hist[i])
        # Rescale y
        all_d = [d for hist in diff_hist for d in hist]
        ax.set_ylim(min(all_d), max(all_d))

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
