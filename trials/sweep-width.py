from contextlib import contextmanager

import numpy as np
import torch
import random
import os
import ff
import S4
from tqdm import tqdm

@contextmanager
def assign_variables(**kwargs):
    """
    Yield the values of kwargs as a tuple, in the same order
    they were passed in (Python 3.6+ preserves kwargs order).

    Usage:
        with assign_variables(x=10, y=20) as (x, y):
            print(x, y)  # 10 20
    """
    values = tuple(kwargs.values())
    try:
        yield values
    finally:
        pass  # no cleanup needed


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
            # Linear interpolation
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

                    x1, y1 = x_values[left_idx], y_values[left_idx]
                    x2, y2 = x_values[right_idx], y_values[right_idx]
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    result.append((x, y))
        else:
            # Polynomial interpolation
            coeffs = np.polyfit(x_values, y_values, actual_poly_order)
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    result.append((x, data_dict[x]))
                else:
                    y = np.polyval(coeffs, x)
                    result.append((x, y))
    else:
        # Just one point, can't interpolate
        result = sorted_data.copy()

    # Extend beyond max_x if requested (always using linear extension)
    if extend > 0 and len(x_values) >= 2:
        second_to_last_x = x_values[-2]
        second_to_last_y = y_values[-2]
        last_x = x_values[-1]
        last_y = y_values[-1]
        slope = (last_y - second_to_last_y) / (last_x - second_to_last_x)
        for i in range(1, extend + 1):
            x = max_x + i
            y = last_y + slope * i
            result.append((x, y))

    return result


def save_checkpoint(path, epoch, gratings, optimizer, grating_epochs):
    checkpoint = {
        'epoch': epoch,
        'gratings': [g.detach().cpu() for g in gratings],
        'optimizer': optimizer.state_dict(),
        'grating_epochs': grating_epochs,
        # NEW ----------------------------
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
    }
    torch.save(checkpoint, path)


'''
This file originally did a 2D sweep over (a,b).  We have replaced that
with a 1D sweep over a single “width” value in [0, 1.1].  Each width is
interpreted as a single, continuous AlN rectangle (the rest of the Grating
layer remains vacuum).  For each width, we log:
  1) the emissivity profile (transmitted_power at each sampled wavelength)
  2) the overall efficiency (FOM)

All other machinery (seeding, material definitions, etc.) is left untouched.
'''

import math
import json
import sys
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────────
#  KEEP ALL OF CONFIG / SEEDING / S4‐SETUP EXACTLY THE SAME UP TO “gradient_per_image”
# ──────────────────────────────────────────────────────────────────────────────────

config = {
    "seeds": {
        "torch": int(92),
        "numpy": int(81),
        "random": int(70)
    },
    "n_incidence_harmonics": int(30),
    "n_grating_harmonics": int(30),
    "n_grating_elements": int(20),   # we won't actually use all 20 “elements” now
    "ang_polarization": float(0),
    "bias": 1e-3,
    "logging": False,    # set True if you want to re‐enable the old logging files
    "optimizer": False
}

if config["logging"]:
    # ... (same logging‐directory setup as before) ...
    pass

torch.manual_seed(config['seeds']['torch'])
np.random.seed(config['seeds']['numpy'])
random.seed(config['seeds']['random'])

wavelengths = torch.linspace(.35, 3, 2651)
exclude_wavelengths = torch.tensor([.5, 1])

y_pt = 0
x_density = 5
n_x_measurement_pts = x_density * config['n_grating_elements']
L = 1.1
z_buff = .15

def make_measurement_grid(L, n_cells, k):
    """
    (unchanged from before)
    """
    dx = L / n_cells
    fractions = np.arange(1, 2 * k, 2) / (2 * k)
    starts = np.arange(n_cells)[:, None]
    return ((starts + fractions) * dx).ravel()

x_measurement_space = make_measurement_grid(L=L, n_cells=x_density, k=config['n_grating_elements'])
depth = .473
z_space = np.linspace(0, .5 + depth + 1, 40)
z_measurement_space = z_space[(z_space >= .5) & (z_space <= .5 + depth)]
jump = 10

# ──────────────────────────────────────────────────────────────────────────────────
#  REPLACED “gradient_per_image” WITH A NEW FUNCTION THAT TAKES A SINGLE WIDTH
# ──────────────────────────────────────────────────────────────────────────────────

def gradient_per_image(width: float, L: float, ang_pol: float, index: int):
    """
    Now interprets “width” as the absolute width of a single AlN rectangle,
    centered horizontally in the Grating layer.  Everything else in the Grating
    is vacuum.

    Returns:
        transmitted_power_list: Python list of transmitted‐power at each sampled wavelength
        fom: scalar torch.Tensor = efficiency
    """
    transmitted_power = []
    indices_used = []
    sample_mask = torch.zeros_like(wavelengths, dtype=torch.bool)

    for i_wl, wl in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
        if wl.item() in exclude_wavelengths or (i_wl % jump) != 0:
            continue

        indices_used.append(i_wl)
        sample_mask[i_wl] = True

        # ----- Setup S4 geometry exactly as before, except:
        #      1) The Grating layer is vacuum by default.
        #      2) We carve out one AlN rectangle of “width” wide (centered).
        S = S4.New(Lattice=L, NumBasis=config['n_grating_harmonics'])
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + 130]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=ff.aln_n[i_wl + 130]**2)

        # Vacuum above, then a “Grating” layer that is all vacuum initially.
        S.AddLayer(Name='VacuumAbove', Thickness=.5, Material='Vac')
        S.AddLayer(Name='Grating',      Thickness=depth, Material='Vac')

        # If width > 0, put exactly one AlN rectangle of that width (centered at L/2)
        if width > 0:
            half_w = width / 2
            S.SetRegionRectangle(
                Layer='Grating',
                Material='AlN',
                Center=(L/2, .5),
                Halfwidths=(half_w, .5),
                Angle=0
            )

        # Add the absorbing tungsten layer underneath
        S.AddLayer(Name='Ab', Thickness=1, Material='W')
        S.SetFrequency(1 / wl)

        # Run the forward solve (no adjoint needed for just efficiency)
        S.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),
            sAmplitude=np.cos(ang_pol * np.pi / 180),
            pAmplitude=np.sin(ang_pol * np.pi / 180),
            Order=0
        )
        (_, back) = S.GetPowerFlux(Layer='VacuumAbove', zOffset=0)
        back_t = torch.as_tensor(back).abs()
        transmitted_power.append(1 - back_t)

        del S

    transmitted_power = torch.tensor(transmitted_power, dtype=torch.float32, requires_grad=False)

    idx = sample_mask.nonzero(as_tuple=True)[0]
    # Compute efficiency (“fom”) exactly as before, using the interpolated data
    # First build a list of (i_idx, transmitted_power[i]) pairs
    data = [(indices_used[i], transmitted_power[i].item()) for i in range(len(indices_used))]
    interpolated_data = interpolate_dataset(data, extend=0, poly_order=1)
    interpolated_ppw = torch.tensor([i[1] for i in interpolated_data], requires_grad=False)

    fom = ff.power_ratio(wavelengths[idx], interpolated_ppw[idx], ff.T_e, .726)

    # If you want to log to the old “fom_file” or “image_file,” you can do it here:
    if config['logging']:
        with open(fom_file, 'a+') as f:
            f.write(f"{index}:{fom.item()}\n")
        with open(image_file, 'a+') as f:
            for val in interpolated_ppw:
                f.write(str(val.item()) + "\n")

    return transmitted_power.tolist(), fom.detach()


# ──────────────────────────────────────────────────────────────────────────────────
#  MAIN:  REPLACE THE 2D (a,b) SWEEP WITH A 1D “width” SWEEP
# ──────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as _np

    # 1) construct a 1D array of widths from 0 to L (1.1), in 0.05‐increments
    width_vals = _np.arange(0.0, L + 1e-6, 0.05)
    Nw = len(width_vals)

    # 2) Check if a previous “width_sweep.npz” exists
    if False:#os.path.exists("width_sweep.npz"):
        data = _np.load("width_sweep.npz", allow_pickle=True)
        effs = data["effs"]                          # shape (Nw,)
        emiss_profiles = data["emiss_profiles"]      # shape (Nw, N_samples)
        width_vals = data["width_vals"]
    else:
        effs = _np.full((Nw,), _np.nan, dtype=_np.float64)
        emiss_profiles = []  # we’ll append a Python list of size‐(N_samples,) each time

    ang_pol = config["ang_polarization"]

    for i, w in enumerate(width_vals):
        # skip if already done
        if not _np.isnan(effs[i]):
            continue

        # Build and solve for this single‐rectangle of width = w
        transmitted_power_list, fom = gradient_per_image(w, L, ang_pol, index=i)

        # 3) Store efficiency
        effs[i] = fom.cpu().item()

        # 4) Store emissivity profile (a Python list of floats)
        emiss_profiles.append(_np.array(transmitted_power_list, dtype=_np.float64))

        # 5) Overwrite the NPZ so far
        #    Since “emiss_profiles” is a Python list of identical‐length arrays, we can stack it.
        emiss_stack = _np.vstack(emiss_profiles)
        _np.savez(
            "width_sweep.npz",
            effs=effs,
            emiss_profiles=emiss_stack,
            width_vals=width_vals
        )

        print(
            f"Swept width={w:.2f} → efficiency={effs[i]:.6f} "
            f"(emiss_profile length={len(transmitted_power_list)})"
        )

    print("Width sweep complete. Data saved to 'width_sweep.npz'.")
