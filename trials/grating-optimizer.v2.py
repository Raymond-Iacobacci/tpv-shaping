from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def assign_variables(**kwargs):
    """
    Yield the values of kwargs as a tuple, in the same order
    they were passed in (Python 3.6+ preserves kwargs order). 

    Usage:
        with assign_variables(x=10, y=20) as (x, y):
            print(x, y)  # 10 20
    """
    # Grab the values in insertion order (guaranteed from Python 3.6+). :contentReference[oaicite:0]{index=0}
    values = tuple(kwargs.values())
    try:
        yield values
    finally:
        # No cleanup needed—locals created by the `as` clause
        # will simply go out of scope when the block ends.
        pass

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


def save_checkpoint(path, epoch, gratings, optimizer, grating_epochs):
    checkpoint = {
        'epoch'          : epoch,
        'gratings'       : [g.detach().cpu() for g in gratings],
        'optimizer'      : optimizer.state_dict(),
        'grating_epochs' : grating_epochs,
        # NEW ----------------------------
        'torch_rng'      : torch.get_rng_state(),
        'cuda_rng'       : torch.cuda.get_rng_state_all(),
        'numpy_rng'      : np.random.get_state(),
        'python_rng'     : random.getstate(),
    }
    torch.save(checkpoint, path)

'''
This file does direct gradient descent on a bilayer system.
The bilayer system has a base of homogeneous Tungsten and a top of a CaO grating, randomly generated at the start.
The bug came from the differing spatial period when used with the ff code. Fix that [ff code].
'''

import json
import os
import random
import secrets
import string
import sys
from pathlib import Path

import ff
import matplotlib.pyplot as plt
import S4
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm

config = {
    "seeds": {
        "torch": int(92),#52
        "numpy": int(81),#41
        "random": int(70)#30
    },
    "n_incidence_harmonics": int(40),
    "n_grating_harmonics": int(40), # >= config['n_incidence_harmonics'], this tells you how many you need to set
    "n_grating_elements": int(22),
    "ang_polarization": float(0), # Better angle for optimization (generally good at showing what can happen in two dimensions with unpolarized metamaterials)
    "learning_rate": float(1e-4),
    "n_gratings": int(1),
    "bias": 1e-3,
    "logging": True,
    "optimizer": False
}
checkpoint = False
if config["logging"]:
    root_log_dir = os.path.join(ff.home_directory(), 'logs', 'gradient-descent-on-grating-t')
    os.makedirs(root_log_dir, exist_ok = True)
    if not checkpoint:
        log_dir = ff.get_unique_log_dir(root_log_dir, config)
        checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
        start_epoch = 0
        grating_epochs = [0] * config['n_gratings']
    else:
        log_dir = "/home/rliacobacci/tpv-shaping/logs/gradient-descent-on-grating-t/def-97060e69_5" # NOTE: THIS NEEDS TO BE ENTERED
        checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
        chk = torch.load(checkpoint_path)

        torch.set_rng_state(chk['torch_rng'])
        torch.cuda.set_rng_state_all(chk['cuda_rng'])
        np.random.set_state(chk['numpy_rng'])
        random.setstate(chk['python_rng'])

        start_epoch = chk['epoch'] + 1
        gratings = [torch.nn.Parameter(torch.empty_like(t)) for t in chk['gratings']]
        for p, saved in zip(gratings, chk['gratings']):
            p.data.copy_(saved)          # keep the same object, just copy the numbers
        grating_epochs = chk['grating_epochs']

    print(f'Log directory: {log_dir}')
    config_file = os.path.join(log_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    fom_file = os.path.join(log_dir, 'fom_values.txt')
    image_file = os.path.join(log_dir, 'image_values.txt')
   
if not checkpoint:
    torch.manual_seed(config['seeds']['torch'])
    np.random.seed(config['seeds']['numpy'])
    random.seed(config['seeds']['random'])

wavelengths = torch.linspace(.35, 3, 2651)
exclude_wavelengths = torch.tensor([.5, 1])

y_pt = 0
x_density = 5
n_x_measurement_pts = x_density * config['n_grating_elements'] # Measuring one x-point per square
L=1.1
z_buff = .15
def make_measurement_grid(L, n_cells, k):
    """
    Return the x–coordinates of k measurements per cell.

    Parameters
    ----------
    L        : total length
    n_cells  : number of unit cells (your n_x_measurement_pts)
    k        : points per cell (k=1 → 0.5; k=2 → 0.25,0.75; k=3 → 1/6,1/2,5/6 …)
    """
    dx        = L / n_cells                               # cell width
    fractions = np.arange(1, 2*k, 2) / (2*k)              # shape (k,)
    starts    = np.arange(n_cells)[:, None]               # shape (n_cells,1)
    return ((starts + fractions) * dx).ravel()            # flatten to 1-D
# x_density = int(n_x_measurement_pts / config['n_grating_elements'])
x_measurement_space = make_measurement_grid(L = L, n_cells = x_density, k = config['n_grating_elements'])
z_space = np.linspace(0, .5 + 1.6 + .473 + 1, 40)
z_measurement_space = z_space[(z_space >= .5) & (z_space <= .5 + 1.6)] # Takes only the relevant items

def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float, index: int) -> torch.Tensor:

    transmitted_power = []
    indices_used = []
    sample_mask = torch.zeros_like(wavelengths, dtype=torch.bool)
    dfom_deps = torch.zeros((config['n_grating_elements']))
    dflux_deps = torch.zeros((wavelengths.shape[0], n_x_measurement_pts))

    for i_wl, wl in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave = False)):
        if wl.item() in exclude_wavelengths or i_wl % 2000 != 0:
            continue
        indices_used.append(i_wl)
        sample_mask[i_wl] = True
        S = S4.New(Lattice = L, NumBasis = config['n_grating_harmonics'])
        S.SetMaterial(Name = 'W', Epsilon = ff.w_n[i_wl]**2)
        S.SetMaterial(Name = 'Vac', Epsilon = 1)
        S.SetMaterial(Name = 'AlN', Epsilon = ff.aln_n[i_wl]**2)

        S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vac')
        S.AddLayer(Name = 'Grating', Thickness = .473, Material = 'Vac')
        
        for ns in range(grating.shape[-1]):
            S.SetMaterial(Name = f'sq{ns+1}', Epsilon = grating[ns].item() * (ff.cao_n[i_wl]**2-1)+1)
            S.SetRegionRectangle(Layer = 'Grating', Material = f'sq{ns+1}', Center = (((ns+1)/grating.shape[-1] - 1/(2*grating.shape[-1]))*L, .5*L), Halfwidths = ((1/(2*grating.shape[-1])*L), .5*L), Angle = 0)

        S.AddLayer(Name = 'Ab', Thickness = 1, Material = 'W')
        S.SetFrequency(1 / wl)

        S_adj = S.Clone()

        S.SetExcitationPlanewave(IncidenceAngles = (0, 0), sAmplitude=np.cos(ang_pol*np.pi/180), pAmplitude=np.sin(ang_pol*np.pi/180), Order=0)
        (_, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0) # We don't need to set the zOffset to the z_buff value because the power flux is the same through both points. The angle is different, yes, but that is not what is being measured in this line.
        back_t = torch.as_tensor(back).abs()
        transmitted_power.append(1 - back_t)

        adj_fields = np.zeros((1, 1, n_x_measurement_pts), dtype = complex)
        for ix, x in enumerate(x_measurement_space):
            adj_fields[0, 0, ix] = S.GetFields(x, y_pt, z_buff)[0][1] # Electric field in the z-direction by RHR at the source point
        grating_fields = np.zeros((z_measurement_space.shape[0], 1, x_measurement_space.shape[0], 3), dtype = complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                grating_fields[iz, 0, ix] = S.GetFields(x, y_pt, z)[0] # Gets electric part of light pointing in three directions to get total intensity at points
        del S

        adj_excitation_mag = np.abs(adj_fields[0, 0, :]**2)
        adj_excitation_phase = np.angle(adj_fields[0, 0, :])
        adj_excitation = adj_excitation_mag * np.exp(1j * adj_excitation_phase)

        fourier_adj_excitation = ff.create_step_excitation(basis = S_adj.GetBasisSet(), step_values = adj_excitation, num_harmonics = config['n_incidence_harmonics'], x_shift = 0, initial_phase = 0, amplitude = 1.0, period = L, plot_fourier = False) # Need to confirm that the polarization is indeed 'x'. I think that it might be 'y'. NOTE: I've set it to 'y' in the ff.py script-file.

        S_adj.SetExcitationExterior(Excitations = tuple(fourier_adj_excitation))

        adj_grating_fields = np.zeros((z_measurement_space.shape[0], 1, x_measurement_space.shape[0], 3), dtype = complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                adj_grating_fields[iz, y_pt, ix] = S_adj.GetFields(x, y_pt, z)[0]
        dflux_deps_wl = 2 * (wl/1e6) ** -2 * ff.e_0 * torch.real(torch.einsum('ijkl,ijkl->ijk', torch.as_tensor(grating_fields), torch.as_tensor(adj_grating_fields).conj())) # NOTE: fix from last time where we squared the wavelength. We are calculating frequency, not wavelength.
        dflux_deps[i_wl] = torch.mean(dflux_deps_wl, dim = 0).squeeze() * (ff.cao_n[i_wl]**2 - 1)
        del S_adj
    # print(len(transmitted_power))
    dflux_deps = torch.tensor(dflux_deps, requires_grad = True) # Add in the gradient to start the backpropagation step

    dfom_deps = dflux_deps[2000].reshape(-1, x_density).mean(axis=1).detach()
    # sys.exit(1)

    # dfom_dflux = interpolated_ppw.grad
    # G = dfom_dflux[idx].unsqueeze(1) * dflux_deps[idx]
    # dwl = 80 * (wavelengths[1] - wavelengths[0])
    # dfom_deps = torch.mean(G, dim = 0) * dwl
    # dfom_deps = dfom_deps.reshape(-1, x_density).mean(axis=1).detach()
    # dfom_deps += config['bias'] * (0.5 - grating) ** 2
    print(f'Index: {index}, previous efficiency: {transmitted_power[1]}')
    print(f'Grating after epoch {epoch+1:03d}: {[round(i.item()*1000)/1000 for i in grating]}')
    # if config['logging']:
    #     with open(fom_file, 'a+') as f:
    #         f.write(str(f'{index}:{fom.item()}\n'))
    #     with open(image_file, 'a+') as f:
    #         for i in grating:
    #             f.write(str(f'{i.item()} '))
    #             f.write("\n")

    return dfom_deps


n_gratings = config['n_gratings']
n_epochs = 1000
if not config['optimizer']:

    gratings = [torch.rand((config['n_grating_elements'],),) for _ in range(n_gratings)]
    # gratings[0][:] = 0.9

    for epoch in range(start_epoch, n_epochs):
        gradient = torch.zeros(size = (gratings[0].shape[-1],))
        for i in range(n_gratings):
            gradient += gradient_per_image(gratings[i], L, config['ang_polarization'], i)
            # print(f"Gradient: {gradient}")
        gradient /= n_gratings
        for i in range(n_gratings):
            gratings[i] -= gradient * config.get('learning_rate', 1e-3) / gradient.norm()
        with torch.no_grad():
            for param in gratings:
                param.data.clamp_(0.0, 1.0)

else:
    if not checkpoint:
        gratings = [torch.nn.Parameter(torch.rand((config['n_grating_elements'],),)) for _ in range(n_gratings)]
        gratings = [torch.rand((config['n_grating_elements'],),) for _ in range(n_gratings)]
        # gratings[0][:] = 1
        # gratings[0][20:] = 0
    opt = torch.optim.Adam(gratings, lr = config.get('learning_rate', 1e-3))
    if checkpoint:
        opt.load_state_dict(chk['optimizer'])
    for epoch in range(start_epoch, n_epochs):
        opt.zero_grad()
        grads = []
        for i, gr in enumerate(gratings):
            dfom_deps = gradient_per_image(gr, L, config['ang_polarization'], i)
            grads.append(dfom_deps)
            grating_epochs[i] += 1
        
        avg_grad = torch.stack(grads, dim = 0).mean(dim = 0)

        for param in gratings:
            bias_grad = config['bias'] * 2 * (0.5 - param.data)
            param.grad = (avg_grad.clone() + (0 if epoch < 500 else bias_grad))#+ bias_grad)
        opt.step()
        with torch.no_grad():
            for param in gratings:
                param.data.clamp_(0.0, 1.0)

        if config['logging']:
            save_checkpoint(
                checkpoint_path,
                epoch,
                gratings,
                opt,              # your Adam optimizer
                grating_epochs    # list you maintain below
            )