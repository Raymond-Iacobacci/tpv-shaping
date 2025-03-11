"""
This script tests convergence of the emissivity calculation by 
increasing the number of harmonics and plotting the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import S4
import ff  # Assuming this is your custom module

# Use the same wavelength range as in the original code
wavelengths = torch.linspace(.350, 3, 2651)
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)]  # Remove problematic wavelengths

# Parameters from original simulation
vacuum_thickness = 0.5  # microns
grid_thickness = 0.473
absorb_thickness = 1.0
num_image_squares = 10

# Create a sample image (similar to the original)
image = torch.rand(num_image_squares)

# Harmonic values to test for convergence
harmonic_values = [5, 10, 14, 20]#, 30, 50]

# Store results for each harmonic setting
emissivity_results = {}

for harmonics in harmonic_values:
    print(f"Testing with {harmonics} harmonics...")
    
    # Store results for current harmonic setting
    transmitted_power = np.zeros(len(wavelengths))
    
    for i_wavelength, wavelength in enumerate(wavelengths):
        # Setup simulation with current number of harmonics
        S = S4.New(Lattice=1, NumBasis=harmonics)
        S.SetOptions(LanczosSmoothing=True)

        # Set materials
        S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)
        S.SetMaterial(Name='Vacuum', Epsilon=(1 + 0j)**2)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)

        # Add layers
        S.AddLayer(Name='VacuumAbove', Thickness=vacuum_thickness, Material='Vacuum')
        S.AddLayer(Name='Grid', Thickness=grid_thickness, Material='Vacuum')

        # Add patterned regions based on image
        for q in range(num_image_squares):
            S.SetMaterial(Name=f'Grid material {q}', 
                          Epsilon=image[q].item() * (ff.aln_n[i_wavelength]**2 - 1) + 1)
            S.SetRegionRectangle(Layer='Grid', 
                                Material=f'Grid material {q}', 
                                Center=((q+1)/num_image_squares - 1/(2*num_image_squares), 0.5),
                                Halfwidths=(1/(2*num_image_squares), 1/2), 
                                Angle=0)

        S.AddLayer(Name='Absorber', Thickness=absorb_thickness, Material='W')

        # Set frequency and excitation
        S.SetFrequency(1 / wavelength)
        S.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),  # Using 0 for incidence angle 
            sAmplitude=1/np.sqrt(2), 
            pAmplitude=1/np.sqrt(2), 
            Order=0
        )

        # Calculate power flux
        (forw, back) = S.GetPowerFlux(Layer='VacuumAbove', zOffset=0)
        transmitted_power[i_wavelength] = 1 - np.abs(back)
    
    # Calculate emissivity (assumes emissivity = absorptivity = 1 - reflectivity)
    emissivity_results[harmonics] = transmitted_power

# Plotting the results
plt.figure(figsize=(12, 8))

for harmonics, emissivity in emissivity_results.items():
    plt.plot(wavelengths, emissivity, label=f'{harmonics} harmonics')

# Remove the problematic blackbody curve line
# plt.plot(wavelengths, ff.T_e, 'k--', label='Blackbody (T=726K)')

plt.xlabel('Wavelength (μm)')
plt.ylabel('Emissivity')
plt.title('Convergence Test: Emissivity vs. Number of Harmonics')
plt.legend()
plt.grid(True)
plt.xlim(0.5, 3)
plt.ylim(0, 1)

# Add a second plot showing the difference between consecutive harmonic settings
plt.figure(figsize=(12, 8))
prev_harmonics = harmonic_values[0]
for harmonics in harmonic_values[1:]:
    difference = np.abs(emissivity_results[harmonics] - emissivity_results[prev_harmonics])
    plt.plot(wavelengths, difference, label=f'Diff: {prev_harmonics} to {harmonics}')
    prev_harmonics = harmonics

plt.xlabel('Wavelength (μm)')
plt.ylabel('Absolute Difference in Emissivity')
plt.title('Convergence Test: Difference Between Consecutive Harmonic Settings')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale to better see small differences
plt.xlim(0.5, 3)

plt.tight_layout()
plt.show()

# Print the maximum difference between highest and lowest harmonic settings
max_diff = np.max(np.abs(emissivity_results[harmonic_values[-1]] - emissivity_results[harmonic_values[0]]))
print(f"Maximum difference between {harmonic_values[0]} and {harmonic_values[-1]} harmonics: {max_diff:.6f}")
