import numpy as np
from contextlib import contextmanager

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

import os
import sys

import ff
import matplotlib.pyplot as plt
import S4
from tqdm import tqdm
import torch

config = {
    "seeds": {
        "numpy": int(41),
    },
    "incidence_angle": float(0),
    # "image_harmonics": int(361),
    "image_harmonics": int(5),
    "polarization_angle": float(0)
}
base_log_dir = os.path.join(ff.home_directory(), "logs")
os.makedirs(base_log_dir, exist_ok=True)
np.random.seed(config['seeds']['numpy'])

# wavelengths = torch.linspace(.350, 3, 2651)
wavelengths = torch.linspace(.22, 5, 4781)
exclude_wavelengths = torch.tensor([.5, 1.])

spec_dir = f'{base_log_dir}/central-dep-concluded-double-symmetric-9'
os.makedirs(spec_dir, exist_ok=True)

L = 1.1

transmitted_power_per_wavelength = []
indices_used = []
for i_wavelength, wavelength in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
# with assign_variables(i_wavelength=10, wavelength=.360) as (i_wavelength, wavelength):
    # if wavelength.item() in exclude_wavelengths or (i_wavelength % 2600 != 0) or i_wavelength == 0:
    # if i_wavelength < 130:
        # continue
    indices_used.append(i_wavelength)
    '''
    S = S4.New(Lattice = ((L, 0), (0, L)), NumBasis=config['image_harmonics'])

    S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
    S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)    # Simple approximate usage
    S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)
    S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')
    # S.AddLayer(Name = 'Grid0', Thickness = .5, Material = 'Vacuum')
    
    for i_grid in range(1, 12):
        material = 'W' if i_grid % 2 == 1 else 'AlN'
        S.AddLayer(Name = f'Grid{i_grid}', Thickness = .015, Material = 'Vacuum')
        S.SetRegionCircle(Layer = f'Grid{i_grid}', Material = material, Center = (.5*L, .5*L), Radius = .12/.4*L)
        S.SetRegionCircle(Layer = f'Grid{i_grid}', Material = 'Vacuum', Center = (.5*L, .5*L), Radius= .06/.4*L)
        # S.SetRegionCircle(Layer = f'Grid{i_grid}', Material = 'W' if i_grid % 2 == 1 else 'AlN', Center = (.5*L, .5*L), Radius = .05*L)
    '''

    S = S4.New(Lattice = (L), NumBasis=config['image_harmonics'])

    S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
    S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)    # Simple approximate usage
    S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)
    S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')

    S.AddLayer(Name = 'Substrate', Thickness = .473, Material = 'AlN')
    S.AddLayer(Name = 'Absorber', Thickness = 1, Material = 'W')
    S.SetFrequency(1 / wavelength)
    S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0)
    (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
    transmitted_power_per_wavelength.append(1 - np.abs(back))
    print(f'Transmitted power at {wavelength} um: {1 - np.abs(back)}')
    '''
    zspace = np.linspace(0, .5+.9+.473+1, 50)
    yspace = np.linspace(0, L, 50)
    xspace = np.linspace(0, L, 50)
    eps_map_reconstructed = np.zeros((len(zspace), len(xspace)), dtype = complex)

    for zi, z in enumerate(zspace):
        for xi, x in enumerate(xspace):
            eps_map_reconstructed[zi, xi] = S.GetEpsilon(x, .25*L, z)

    levels = np.linspace(np.min(np.real(eps_map_reconstructed)), np.max(np.real(eps_map_reconstructed)), 10)
    plt.contourf(xspace, zspace, np.real(eps_map_reconstructed), levels = levels, cmap = 'viridis')
    plt.colorbar()
    plt.title('Y-cross section (real part of epsilon)')
    plt.show()

    levels = np.linspace(np.min(np.imag(eps_map_reconstructed)), np.max(np.imag(eps_map_reconstructed)), 10)
    plt.contourf(xspace, zspace, np.imag(eps_map_reconstructed), levels = levels, cmap = 'viridis')
    plt.colorbar()
    plt.title('Y-cross section (imaginary part of epsilon)')
    plt.show()

    for zi, z in enumerate(zspace):
        for yi, y in enumerate(yspace):
            eps_map_reconstructed[zi, yi] = S.GetEpsilon(.5*L,y,z)

    levels = np.linspace(np.min(np.real(eps_map_reconstructed)), np.max(np.real(eps_map_reconstructed)), 10)
    plt.contourf(xspace, zspace, np.real(eps_map_reconstructed), levels = levels, cmap = 'viridis')
    plt.colorbar()
    plt.title('X-cross section (real part of epsilon)')
    plt.show()

    levels = np.linspace(np.min(np.imag(eps_map_reconstructed)), np.max(np.imag(eps_map_reconstructed)), 10)
    plt.contourf(xspace, zspace, np.imag(eps_map_reconstructed), levels = levels, cmap = 'viridis')
    plt.colorbar()
    plt.title('X-cross section (imaginary part of epsilon)')
    plt.show()
    '''
    '''
    S.AddLayer(Name = 'Substrate', Thickness = .04, Material = 'AlN')
    S.AddLayer(Name = 'Absorber', Thickness = .12, Material = 'W')
    S.SetFrequency(1 / wavelength)
    S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0)
    (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
    transmitted_power_per_wavelength.append(1 - np.abs(back))
    '''
    # print(f"Transmitted power at {wavelength}: {1 - np.abs(back)}")
    # with open(f'{spec_dir}/transmitted_power.txt', 'a') as f:
    #     f.write(f"{wavelength.item()}: {1 - np.abs(back)}\n")
    if np.isnan(back):
        print(f"Transmitted power at {wavelength.item()}: {1 - np.abs(back)}")

# sys.exit(1)

# data = [(indices_used[i], transmitted_power_per_wavelength[i]) for i in range(len(indices_used))]
# interpolated_data = interpolate_dataset(data, extend=60, poly_order=1)
# interpolated_ppw = torch.tensor([i[1] for i in interpolated_data])
interpolated_ppw = torch.tensor(transmitted_power_per_wavelength)

bandgaps = [.5, .6, .61, .62, .63, .64, .65, .66, .67, .68, .69, .7, .71, .72, .726, .8, .9, 1]

FOMs = []

for bandgap in bandgaps:
    FOMs.append(ff.power_ratio(wavelengths, interpolated_ppw, ff.T_e, bandgap).item())
    print(f"Bandgap: {bandgap}, FOM: {FOMs[-1]}")

plt.plot(interpolated_ppw.numpy())
plt.show()

zspace = np.linspace(0, .5+.9+.473+1, 50)
xspace = np.linspace(0, L, 50)
eps_map_reconstructed = np.zeros((len(zspace), len(xspace)), dtype = complex)

for zi, z in enumerate(zspace):
    for xi, x in enumerate(xspace):
        eps_map_reconstructed[zi, xi] = S.GetEpsilon(x, .5, z)

levels = np.linspace(np.min(np.real(eps_map_reconstructed)), np.max(np.real(eps_map_reconstructed)), 10)
plt.contourf(xspace, zspace, np.real(eps_map_reconstructed), levels = levels, cmap = 'viridis')
plt.colorbar()
plt.title('Real part of epsilon')
plt.show()

levels = np.linspace(np.min(np.imag(eps_map_reconstructed)), np.max(np.imag(eps_map_reconstructed)), 10)
plt.contourf(xspace, zspace, np.imag(eps_map_reconstructed), levels = levels, cmap = 'viridis')
plt.colorbar()
plt.title('Imaginary part of epsilon')
plt.show()

sys.exit(1)

with open(f'{spec_dir}/setup.txt', "a+") as f:
    f.write("""
    L = 1.1
    S = S4.New(Lattice = ((L, 0), (0, L)), NumBasis=config['image_harmonics'])

    S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
    S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)    # Simple approximate usage
    S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)
    S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')
    # S.AddLayer(Name = 'Grid0', Thickness = .5, Material = 'Vacuum')
    S.AddLayer(Name = 'Grid', Thickness = .9, Material = 'Vacuum') #2.2

    # S.SetRegionRectangle(Layer = 'Grid',  Material = f'W', Center = (.75*L, .5*L), Halfwidths = (.05*L, .5*L), Angle = 0)
    S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.5*L, .0*L), Halfwidths = (.1*L, .1*L), Angle = 0)
    S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.5*L, .5*L), Halfwidths = (.1*L, .1*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid0', Material = f'AlN', Center = (.5*L, .5*L), Halfwidths = (.1*L, .1*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.25*L, .5*L), Halfwidths = (.05*L, .5*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.5*L, .75*L), Halfwidths = (.05*L, .5*L), Angle = 0)
    S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.5*L, .25*L), Halfwidths = (.05*L, .5*L), Angle = 0)
    S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.5*L, .75*L), Halfwidths = (.05*L, .5*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.25*L, .5*L), Halfwidths = (.5*L, .05*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.75*L, .5*L), Halfwidths = (.5*L, .05*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.25*L, .5*L), Halfwidths = (.5*L, .05*L), Angle = 0)
    # S.SetRegionRectangle(Layer = 'Grid', Material = f'AlN', Center = (.25*L, .25*L), Halfwidths = (.05*L, .05*L), Angle = 0)

    S.AddLayer(Name = 'Substrate', Thickness = .473, Material = 'AlN')
    S.AddLayer(Name = 'Absorber', Thickness = 1, Material = 'W')
    S.SetFrequency(1 / wavelength)
    S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0)
    (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
    transmitted_power_per_wavelength.append(1 - np.abs(back))
#     """)

torch.save(interpolated_ppw, f'{spec_dir}/emit.pt')

with open(f'{spec_dir}/foms.txt', 'w') as f:
    for fom in FOMs:
        f.write(f'{fom}\n')