# TODO: change correction k-vector for each nonzero harmonic
# TODO: figure out why adding in evanescent modes makes it distance-dependent long past where they should have decayed
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

def interpolate_dataset(data, extend=0, poly_order=1):
    """
    Interpolate a dataset of (x,y) pairs, where x are integers, using polynomial interpolation.
    
    Args:
        data: List of (x,y) tuples where x are integers
        extend: Number of points to extend beyond max(x)
        poly_order: Order of the polynomial to use for interpolation (1=linear, 2=quadratic, etc.)
                    Will automatically reduce order if not enough points are available
    
    Returns:
        List of (x,y) tuples with interpolated and extended values
    """
    # Sort data by x values
    sorted_data = sorted(data, key=lambda point: point[0])
    
    # Extract x and y values
    x_values = [point[0] for point in sorted_data]
    y_values = [point[1] for point in sorted_data]
    
    # Find min and max x values
    min_x = min(x_values)
    max_x = max(x_values)
    
    # Create a dictionary from original data for easy lookup
    data_dict = {x: y for x, y in sorted_data}
    
    # Ensure we have enough points for the polynomial order
    actual_poly_order = min(poly_order, len(sorted_data) - 1)
    if actual_poly_order != poly_order:
        print(f"Warning: Reduced polynomial order from {poly_order} to {actual_poly_order} due to insufficient data points")
    
    result = []
    
    # If we have enough points for polynomial interpolation
    if len(sorted_data) > 1:
        if actual_poly_order == 1:
            # Linear interpolation (original method)
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    result.append((x, data_dict[x]))
                else:
                    # Find closest points before and after x
                    left_idx = 0
                    while left_idx < len(x_values) - 1 and x_values[left_idx + 1] <= x:
                        left_idx += 1
                    
                    right_idx = len(x_values) - 1
                    while right_idx > 0 and x_values[right_idx - 1] >= x:
                        right_idx -= 1
                    
                    # Interpolate y value
                    x1, y1 = x_values[left_idx], y_values[left_idx]
                    x2, y2 = x_values[right_idx], y_values[right_idx]
                    
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    result.append((x, y))
        else:
            # Polynomial interpolation
            coeffs = np.polyfit(x_values, y_values, actual_poly_order)
            
            # Interpolate for all integer points
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    # Use original data point if available
                    result.append((x, data_dict[x]))
                else:
                    # Use polynomial to interpolate
                    y = np.polyval(coeffs, x)
                    result.append((x, y))
    else:
        # Just one point, can't interpolate
        result = sorted_data.copy()
    
    # Extend beyond max_x if requested (always using linear extension)
    if extend > 0 and len(x_values) >= 2:
        # Get the last two points to determine the slope for extension
        second_to_last_x = x_values[-2]
        second_to_last_y = y_values[-2]
        last_x = x_values[-1]
        last_y = y_values[-1]
        
        # Calculate slope
        slope = (last_y - second_to_last_y) / (last_x - second_to_last_x)
        
        # Add extended points
        for i in range(1, extend + 1):
            x = max_x + i
            y = last_y + slope * i
            result.append((x, y))
    
    return result

config = {
    "seeds": {
        "torch": int(92),
        "numpy": int(81),
    },
    "n_grating_harmonics": int(30), # >= config['n_incidence_harmonics'], this tells you how many you need to set
    "n_grating_elements": int(20),
    "ang_polarization": float(0), # Better angle for optimization (generally good at showing what can happen in two dimensions with unpolarized metamaterials)
    "learning_rate": float(1),
    "n_gratings": int(1),
    "logging": False,
}

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
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float,depth: float,
                       plot_fields: bool = False):
    p = 20
    n_grating_elements = grating.shape[-1]
    x_density = 5
    n_x_pts = x_density * n_grating_elements
    # depth = 0.7
    vac_depth = 0.00
    exclude_wl = [.5, 1]
    jump = 80
    z_meas = np.linspace(vac_depth, vac_depth + depth, 30)
    wavelengths = torch.linspace(0.35, 3.0, 2651)
    z_buf = 0. # irrespective to this in homogeneous case but...

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2 * k, 2) / (2 * k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()

    x_space = make_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux = torch.zeros((len(wavelengths), n_x_pts))
    power = []
    N = 13
    indices_used = []
    wl_mask = torch.zeros_like(wavelengths, dtype=torch.bool)
    for i_wl, wl in enumerate(tqdm(wavelengths, desc="Wavelengths", leave = False)):
        if wl.item() in exclude_wl or i_wl % jump != 0:
            continue
        indices_used.append(i_wl)
        wl_mask[i_wl] = True
        S = S4.New(Lattice=L, NumBasis=N)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + 130]**2)

        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl + 130]**2 - 1) * grating[0].item() + 1)

        S.AddLayer(Name='VacuumAbove', Thickness=vac_depth, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='Vac')
        S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (L/4, L/2), Angle = 0)
        for ns in range(len(grating)):
            S.SetMaterial(Name = f'SquareMat-{ns+1}', Epsilon = grating[ns].item() * (ff.aln_n[i_wl+130]**2-1)+1)
            S.SetRegionRectangle(Layer = 'Grating', Material = f'SquareMat-{ns+1}', Center = (((ns+1) / len(grating) - 1 / (2 * len(grating))) * L, .5), Halfwidths = ((1 / (2 * len(grating))*L), .5), Angle = 0)
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave((0, 0),
                                 sAmplitude=np.cos(ang_pol * np.pi/180),
                                 pAmplitude=np.sin(ang_pol * np.pi/180),
                                 Order=0)
        forw, back = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(1-np.abs(back))
        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buf)

        fwd_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                fwd_meas[iz, 0, ix] = S.GetFields(x, 0, z)[0]

        k0 = 2 * np.pi / wl.item()

        S_adj = S.Clone()

        basis = S_adj.GetBasisSet() # Removes the repeated calls
        
        pos_harmonics = [ i for i in range(len(basis)) if basis[i][0] > 0 and abs(2*np.pi*basis[i][0]/L) <= k0]
        neg_harmonics = [ i for i in range(len(basis)) if basis[i][0] <= 0 and abs(2*np.pi*basis[i][0]/L) <= k0]
        k0 = 2 * np.pi / wl.item()
        excitations = []
        for i, raw_amp in enumerate(back_amp):
            if i not in pos_harmonics and i not in neg_harmonics:
                continue
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            if i in pos_harmonics:
                excitations.append((2*basis[i][0], b'y', corr_amp))
            if i in neg_harmonics:
                excitations.append((-2*basis[i][0]+1, b'y', corr_amp))
        S_adj.SetExcitationExterior(tuple(excitations))
        adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                adj_meas[iz, 0, ix] = S_adj.GetFields(x, 0, z)[0]
        term = -k0 * torch.imag(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(fwd_meas),
                         torch.as_tensor(adj_meas)) # This rotation factor is equivalent to taking the -imaginary value of the system
        )
        dz = (depth) / len(z_meas)
        dflux[i_wl] = term.mean(dim=0).squeeze() * dz * L / n_x_pts
        dflux[i_wl] = dflux[i_wl] * (ff.aln_n[i_wl+130]**2 - 1) # HERE IT IS!

        if plot_fields:
            z_space = np.linspace(0, 1+vac_depth + depth, 30)
            fwd_vol = np.zeros((z_space.size, n_x_pts), complex)
            adj_vol = np.zeros((z_space.size, n_x_pts), complex)
            for iz, z in enumerate(z_space):
                for ix, x in enumerate(x_space):
                    fwd_vol[iz, ix] = S.GetFields(x, 0, z)[0][1]
                    adj_vol[iz, ix] = S_adj.GetFields(x, 0, z)[0][1] + S_adj.GetFields(-x, 0, z)[0][1]


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

        del S, S_adj
    n_wl, total_pts = dflux.shape

    dflux = torch.tensor(dflux, requires_grad = True)
    power = torch.tensor(power, requires_grad = True)
    power_data = [(indices_used[i], power[i]) for i in range(len(indices_used))]
    interpolated_power = torch.tensor([i[1] for i in interpolate_dataset(power_data, extend = 10, poly_order = 1)], requires_grad = True)
    interpolated_power.retain_grad()

    idx = wl_mask.nonzero(as_tuple = True)[0]
    eff = ff.power_ratio(wavelengths[idx], interpolated_power[idx], ff.T_e, .726)
    eff.backward()

    dflux_dwl = interpolated_power.grad * jump # type: ignore
    # print(dflux.shape)
    deff_dmeas = torch.sum(dflux_dwl[:, None] * dflux, dim = 0)
    total_pts = deff_dmeas.numel()
    if total_pts % len(grating) != 0:
        raise ValueError(f"Inconsistent measured-point counts: {total_pts} points "
                         f"cannot be evenly divided into {len(grating)} squares")
    dflux_dgeo = deff_dmeas.view(len(grating), x_density).sum(dim=1)
    return dflux_dgeo, power, eff.detach()

for depth in [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.1]:
    grating = torch.rand((config['n_grating_elements'],))
    prev_eff = 0
    for epoch in range(150):
        gradient, power, eff = gradient_per_image(grating, 1, 0, depth)
        if eff < prev_eff:
            print("\n\nFAILED\n")
            # break
        prev_eff = eff
        # print('-'*30)
        # print(f'E: {epoch} >> P: {power.detach().numpy()}')
        print(f'Ef: {eff.numpy()}')
        # print(f'G: {gradient.detach().numpy()}')
        # print(f'V: {grating}')
        grating -= gradient
        with torch.no_grad():
            grating.data.clamp_(0.0, 1.0)
    import time
    time.sleep(10)