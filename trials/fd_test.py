#!/usr/bin/env python
# finite_difference_check.py
# -------------------------------------------------
# 1)  All original imports and helper functions
#     (interpolate_dataset, assign_variables, etc.)
# -------------------------------------------------
from contextlib import contextmanager
import numpy as np

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

import json, os, sys
import torch
import S4, ff
from tqdm import tqdm

# ---------- (everything up to 'config = {...}' is identical) ----------
#  ...  < keep your interpolate_dataset, assign_variables, etc. here > ...


# ------------------------ configuration ------------------------------
config = {
    "seeds": {"torch": 52, "numpy": 41},
    "n_incidence_harmonics": 40,
    "n_grating_harmonics"  : 40,
    "n_grating_elements"   : 20,
    "ang_polarization"     : 0.0,   # degrees
}
torch.manual_seed(config["seeds"]["torch"])
np.random.seed(config["seeds"]["numpy"])

# wavelength grid and sampling mask (every 80-th point, skip 0.5 & 1 µm)
wavelengths        = torch.linspace(0.35, 3.0, 2651)
exclude_wavelength = torch.tensor([0.5, 1.0])
sample_mask        = torch.zeros_like(wavelengths, dtype=torch.bool)
sample_mask[::80]  = True
for w in exclude_wavelength: sample_mask &= wavelengths != w
idx = sample_mask.nonzero(as_tuple=True)[0]          # 53 wavelengths
Δλ  = (wavelengths[1]-wavelengths[0]) * 80           # 0.08 µm

# measurement geometry
y_pt   = 0
L      = 1.1
z_buff = 0.15
n_x    = config["n_grating_elements"]
x_pts  = np.linspace(0, L, 2*n_x+1)[1::2]
z_all  = np.linspace(0, .5+1.6+.473+1, 10)
z_pts  = z_all[(z_all >= .5) & (z_all <= .5+1.6)]

# -------------------------------------------------
# 2)  Core routine: returns FOM and adjoint gradient
# -------------------------------------------------
def forward_and_adjoint(p_vec):
    """Compute FOM and ∂F/∂p for a given grating parameter vector (shape 20)."""
    transmitted = []
    dflux_deps  = torch.zeros(len(wavelengths), n_x)

    p_vec = p_vec.detach().cpu()

    for i, wl in enumerate(wavelengths):
        if not sample_mask[i]:                 # skip unused λ
            continue
        S = S4.New(Lattice=L, NumBasis=config["n_grating_harmonics"])
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=ff.aln_n[i]**2)

        S.AddLayer(Name='VacuumAbove', Thickness=.5,   Material='Vac')
        S.AddLayer(Name='Grating',     Thickness=.473, Material='Vac')

        for k in range(n_x):
            eps = p_vec[k]*(ff.cao_n[i]**2-1)+1
            S.SetMaterial(Name=f'sq{k}', Epsilon=eps.item())
            cx = ((k+1)/n_x - 0.5/n_x)*1
            S.SetRegionRectangle('Grating', f'sq{k}',
                                 Center=(cx, .5*L),
                                 Halfwidths=(0.5/n_x, .5*L), Angle = 0)

        S.AddLayer(Name='Wlayer', Thickness=1.0, Material='W')
        S.SetFrequency(1/wl)

        # forward power
        S.SetExcitationPlanewave((0,0),
            sAmplitude=1.0, pAmplitude=0.0, Order=0)
        _, back = S.GetPowerFlux('VacuumAbove', 0)
        transmitted.append(1 - abs(back))

        # fields for adjoint overlap
        adj_flds = np.zeros((1,1,n_x), complex)
        for ix,x in enumerate(x_pts):
            adj_flds[0,0,ix] = S.GetFields(x,y_pt,z_buff)[0][1]

        grat_flds = np.zeros((len(z_pts),1,len(x_pts),3), complex)
        for iz,z in enumerate(z_pts):
            for ix,x in enumerate(x_pts):
                grat_flds[iz,0,ix] = S.GetFields(x,y_pt,z)[0]
        S_adj = S.Clone()
        # adjoint excitation (step profile) – sign handled here
        step = np.abs(adj_flds[0,0,:])**2 * np.exp(1j*np.angle(adj_flds[0,0,:]))
        four = ff.create_step_excitation(S_adj.GetBasisSet(), step,
                                         num_harmonics=config["n_incidence_harmonics"], plot_fourier = False)
        S_adj.SetExcitationExterior(tuple(four))
        adj_grat = np.zeros_like(grat_flds)
        for iz,z in enumerate(z_pts):
            for ix,x in enumerate(x_pts):
                adj_grat[iz,0,ix] = S_adj.GetFields(x,y_pt,z)[0]

        # ∂flux/∂ε (no chain-rule yet)
        dflux = 2*(wl*1e-6)**-2*ff.e_0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(grat_flds),
                         torch.as_tensor(adj_grat).conj()))
        dflux_deps[i] = torch.mean(dflux,0) * (ff.cao_n[i]**2-1)   # ∂F/∂p
        del S, S_adj

    transmitted = torch.tensor(transmitted, dtype=torch.float64)
    # interpolate & FOM
    # data = [(j,transmitted[k]) for j,k in enumerate(idx)]
    data = list(zip(idx.tolist(), transmitted.tolist()))
    interp = interpolate_dataset(data, extend=10, poly_order=1)
    ppw    = torch.tensor([v for _,v in interp], dtype=torch.float64, requires_grad = True)
    fom    = ff.power_ratio(wavelengths[idx], ppw[idx], ff.T_e, .726)
    fom.backward()

    # adjoint inner product
    G = ppw.grad.unsqueeze(1) * dflux_deps[idx]      # df/dflux · dflux/dp
    grad_p = torch.sum(G,0) * Δλ                     ### PATCH ###
    return fom.item(), grad_p                        ### PATCH ###


# ------------------------------------------
# 3)  Finite-difference validation routine
# ------------------------------------------
def finite_difference_test(pixels=(0,19), h=1e-4):
    p0 = torch.rand(n_x)          # random starting design
    F0, g_adj = forward_and_adjoint(p0.clone().requires_grad_(True))

    print(f"\nBase FOM = {F0:.8e}\n")
    for k in pixels:
        p_plus         = p0.clone(); p_plus[k]  += h
        p_minus        = p0.clone(); p_minus[k] -= h
        F_plus,  _     = forward_and_adjoint(p_plus)
        F_minus, _     = forward_and_adjoint(p_minus)
        g_fd           = (F_plus - F_minus) / (2*h)
        rel_err        = abs(g_fd - g_adj[k]) / max(1e-12, abs(g_fd))

        print(f"pixel {k:2d}:  adjoint = {g_adj[k]: .4e}   "
              f"finite-diff = {g_fd: .4e}   "
              f"rel-error = {rel_err:.2%}")

# ------------------ run when executed ------------------
if __name__ == "__main__":
    finite_difference_test()
