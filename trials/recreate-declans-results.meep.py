import numpy as np
import torch

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

import meep as mp
import numpy as np
import torch
from tqdm import tqdm
import ff

# Your existing interpolation helper

# Materials: replace with your actual dispersion data
def epsilon_w(t) -> complex:  return complex(ff.w_n[t+130]**2)  # tungsten
def epsilon_AlN(t) -> complex: return complex(ff.aln_n[t+130]**2)

# Geometry parameters
period     = 1.0                  # grating period
vac_above  = 0.5                  # vacuum buffer above
depth2     = .473           # AlN thickness
substrate  = 1.0                  # tungsten thickness
cell_z     = vac_above + depth2 + substrate + 2.0  # extra PML buffer

# MEEP resolution
resolution = 200  # adjust for accuracy vs. speed

# PML layers (top & bottom in z)
pml_layers = [mp.PML(thickness=1.0, direction = mp.Z)]  # absorb ±z :contentReference[oaicite:5]{index=5}

# Simulation cell: x=period (periodic), z as above
cell = mp.Vector3(period, 0, cell_z)

# Plane‐wave source parameters
def make_source(wl):
    fcen  = 1.0 / wl.item()
    src_z = -0.5*cell_z + vac_above/2
    return [ mp.Source(mp.ContinuousSource(frequency=fcen),
                       component=mp.Ez,
                       center=mp.Vector3(0,0,src_z),
                       size=mp.Vector3(period,0,0)) ]

# --- adjust cell height to remove the depth1 “grating” layer ---
cell_z = vac_above + depth2 + substrate + 2.0  # no depth1 in sum any more

# --- rebuild geometry without the middle “grating” block ---
def make_geometry(i_wl, freq):
    w_eps   = epsilon_w(i_wl)
    aln_eps = epsilon_AlN(i_wl)
    real_aln_eps = np.real(aln_eps)
    real_w_eps = np.real(w_eps)
    imag_w_eps = np.imag(w_eps)
    sigma = 2 * np.pi * freq * imag_w_eps / real_w_eps
    geom = []

    # Vacuum above
    geom.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, vac_above),
        center=mp.Vector3(0,0, +0.5*cell_z - 0.5*vac_above),
        material=mp.Medium(epsilon=1.0)
    ))

    # AlN layer (formerly second block)
    geom.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, depth2),
        center=mp.Vector3(0,0, +0.5*cell_z - vac_above - 0.5*depth2),
        material=mp.Medium(epsilon=real_aln_eps)
    ))

    # Tungsten substrate
    geom.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, substrate),
        center=mp.Vector3(0,0, -0.5*cell_z + 0.5*substrate),
        material=mp.Medium(epsilon=real_w_eps,
        D_conductivity = sigma) # NOTE: could be a problem here
    ))
    return geom
# Frequency sampling (single frequency per run here)
nfreq = 1

# Monitors just outside PML (offset by dpml=1.0)
dpml     = 1.0
flux_z_in  = +0.5*cell_z - dpml - 0.1
flux_z_out = -0.5*cell_z + dpml + 0.1

refl_fr = mp.FluxRegion(center=mp.Vector3(0,0, flux_z_in),  size=mp.Vector3(period,0,0))
tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,flux_z_out), size=mp.Vector3(period,0,0))

transmitted = []
indices_used = []

wavelengths = np.linspace(.35, 3, 2651)
jump = 10
for i_wl, wl in enumerate(tqdm(wavelengths, desc="λ sweep")):
    if i_wl % jump != 0:
        continue
    indices_used.append(i_wl)

    # Build sim
    sim = mp.Simulation(
        cell_size       = cell,
        boundary_layers = pml_layers,
        geometry        = make_geometry(i_wl, 1 / (wl * 1e-6)),
        sources         = make_source(wl),
        resolution      = resolution
    )

    # Add flux monitors
    refl = sim.add_flux(1.0/wl.item(), 0, nfreq, refl_fr)
    tran = sim.add_flux(1.0/wl.item(), 0, nfreq, tran_fr)

    cycles = 20                     # 20 field periods
    f = 1 / wl.item()               # frequency
    T = 1 / f                       # period
    sim.run(until=T * cycles)

    # Get flux data

    # Get the complex Ez on the x–z plane (periodic in x, thickness in z)
    ez_data = sim.get_array(component=mp.Ez,
                            center=mp.Vector3(),              # x=0, z=0 slice
                            size=mp.Vector3(period, 0, cell_z))

    # Compute intensity |Ez|^2
    intensity = np.abs(ez_data)**2

    # PLOT THE INTENSITY PROFILE
    import matplotlib.pyplot as plt

    # If you want a 1D cut through z at x=0:
    # plt.figure(figsize=(6,3))
    # plt.plot(intensity[int(intensity.shape[0]//2), :])
    # plt.xlabel("Grid index (z)")
    # plt.ylabel(r"$|E_z|^2$")
    # plt.title(f"Ez intensity (λ = {wl:.3f})")
    # plt.tight_layout()
    # plt.show()

    # Optionally, to see the entire x–z field:
    plt.figure(figsize=(6,4))
    plt.imshow(intensity.T, origin='lower', aspect='auto')
    plt.colorbar(label=r"$|E_z|^2$")
    plt.xlabel("x grid index")
    plt.ylabel("z grid index")
    plt.title(f"Ez |^2 (λ = {wl:.3f})")
    plt.show()

    back_flux = abs(mp.get_fluxes(refl)[0])
    transmitted.append(1 - back_flux)
    sim.reset_meep()
    print(f'Transmitted: {transmitted[-1]}')

# Interpolate & convert to tensor
data = list(zip(indices_used, transmitted))
interpolated = interpolate_dataset(data, extend=0, poly_order=1)
emissivity = torch.tensor([y for x,y in interpolated], requires_grad=True)
