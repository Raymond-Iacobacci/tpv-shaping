#!/usr/bin/env python3
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

import torch
import matplotlib.pyplot as plt
import ff    # your material database module
import S4    # S4 scattering engine

def compute_emissivity_spectrum(cao_thickness, wavelengths, exclude_wavelengths, config):
    """
    Build a three-layer S4 structure:
      - Vacuum above (semi-infinite)
      - CaO film of given thickness
      - Tungsten substrate (thick enough to suppress transmission)
    Returns emissivity(λ) = 1 – reflectance(λ).
    """
    L = config['L']                   # lateral period (set to 1 μm here)
    eps_vac = 1.0

    # sweep frequencies
    emissivity = []
    indices_used = []
    for i, wl in enumerate(wavelengths):
        if wl.item() in exclude_wavelengths or i % 40 != 0:
            continue
        eps_cao = ff.cao_n[i]**2            # ff.cao_n is an array over wavelengths
        eps_w   = ff.w_n[i+130]**2
        eps_mgo = ff.mgo_n[i]**2
        eps_aln = ff.aln_n[i+130]**2

        S = S4.New(Lattice=L, NumBasis=config['n_grating_harmonics'])
        # define materials
        S.SetMaterial(Name='Vac', Epsilon=eps_vac)
        S.SetMaterial(Name='CaO', Epsilon=eps_cao)
        S.SetMaterial(Name='W',   Epsilon=eps_w)
        S.SetMaterial(Name='MgO', Epsilon=eps_mgo)
        S.SetMaterial(Name='AlN', Epsilon=eps_aln)

        # layers: name, thickness (μm), material
        S.AddLayer(Name='VacAbove', Thickness=.5, Material='Vac')    # semi-infinite top
        S.AddLayer(Name='Film2',    Thickness=cao_thickness/2, Material='CaO')
        # S.AddLayer(Name='Grid',     Thickness=.5, Material='W')
        S.AddLayer(Name='Film',     Thickness=cao_thickness/2, Material='CaO')
        S.AddLayer(Name='Sub',      Thickness=1.0, Material='W')      # thick W substrate

        indices_used.append(i)
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),
            sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2),
            Order=0
        )
        # get forward/backward flux in the top vacuum
        _, back = S.GetPowerFlux(Layer='VacAbove', zOffset=0)
        R = torch.as_tensor(back).abs()           # reflectance
        emissivity.append(1.0 - R)                   # Kirchhoff: ε = 1 – R
    return emissivity,indices_used

def main():
    # -------- configuration and wavelength grid --------
    config = {
        'n_grating_harmonics': 40,
        'L': 1.0    # lateral period in μm
    }
    wavelengths = torch.linspace(0.35, 3.0, 2651)  # μm
    exclude_wavelengths = torch.tensor([.5, 1])

    # -------- sweep CaO thicknesses --------
    n_points = 30
    depths = np.linspace(1.0, 0.01, n_points)  # μm
    all_emissivities = []

    print("Depth (μm)   FOM")
    for depth in depths:
        ε, indices_used = compute_emissivity_spectrum(depth, wavelengths, exclude_wavelengths, config)
        data = [(indices_used[i], ε[i]) for i in range(len(indices_used))]
        interpolated_data = interpolate_dataset(data, extend = 10, poly_order = 1)
        interpolated_ε = torch.tensor([i[1] for i in interpolated_data], requires_grad = True)
        all_emissivities.append(interpolated_ε)

        # compute FOM: power ratio against blackbody at T_e with emissivity ε
        fom = ff.power_ratio(wavelengths, interpolated_ε, ff.T_e, 0.726)
        print(f"{depth:8.3f}   {fom.item():.6f}")

    # -------- plot every 10th emissivity --------
    plt.figure(figsize=(8, 6))
    for idx in range(0, n_points, 10):
        plt.plot(
            wavelengths.numpy(),
            all_emissivities[idx].detach().numpy(),
            label=f"{depths[idx]:.2f} μm"
        )
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Emissivity")
    plt.title("Emissivity Spectra for Every 10th CaO Thickness")
    plt.legend(title="CaO thickness")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
