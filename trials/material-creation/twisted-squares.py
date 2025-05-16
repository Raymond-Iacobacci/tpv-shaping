"""
This file simulates trials of random structures with different parameters
"""


import json
import os
import sys
import time
from pathlib import Path

import ff
import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
from tqdm import tqdm
import gc

config = {
    "incidence_angle": float(0),
    "image_harmonics": int(400),
    "polarization_angle": float(45),
}

base_log_dir = os.path.join(ff.home_directory(), 'logs')
os.makedirs(base_log_dir, exist_ok=True)

log_dir = ff.get_unique_log_dir(base_log_dir, config)
config_file = os.path.join(log_dir, 'config.json')
with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)


fom_file = os.path.join(log_dir, 'fom_values.txt')

wavelengths = torch.linspace(.350, 3, 2651)  # Issue when this goes past 99?

# wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)] # This is necessary due to S4 bugging out at these wavelengths

for i in range(20):

    transmitted_power_per_wavelength = torch.zeros((len(wavelengths),))
    forward_power_per_wavelength = torch.zeros((len(wavelengths),))
    np.random.seed(i)
    layers = 2
    initialization_params = np.random.rand(2 * layers)
    thicknesses, widths = initialization_params[:layers], initialization_params[layers:]
    np.sort(widths)
    print(f'Thicknesses, widths: {thicknesses}, {widths}')
    for i_wavelength, wavelength in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
        if i_wavelength % 20 != 0:
            continue
        if wavelength == 0.5 or wavelength == 1.0:
             continue
        L = .3
        S = S4.New(Lattice = ((L, 0), (0,L)), NumBasis=config['image_harmonics'])
        S.SetOptions(LanczosSmoothing=True)
        S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
        S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)    # Simple approximate usage
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)
        S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')

        S.AddLayer(Name = f'Grid0', Thickness = thicknesses[0], Material = 'Vacuum')
        # S.SetRegionCircle(Layer = 'Grid0', Material = f'W', Center = (.3, .3), Radius = widths[0] * .3)
        S.SetRegionRectangle(Layer = 'Grid0', Material = f'W', Center = (L/2, L/2), Halfwidths = (widths[0]/2*L, widths[0]/2*L), Angle = 0)
        S.AddLayer(Name = f'Grid1', Thickness = thicknesses[1], Material = 'Vacuum')
        # S.SetRegionCircle(Layer = 'Grid1', Material = f'AlN', Center = (.3, .3), Radius = radii[1] * .3)
        S.SetRegionRectangle(Layer = 'Grid1', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (widths[1]/2/np.sqrt(2)*L, widths[1]/2/np.sqrt(2)*L), Angle = 45)
        S.AddLayer(Name = 'VacuumBelow', Thickness = .5, Material = 'Vacuum')

        S.SetFrequency(1 / wavelength)
        S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0) # In the real simulation this should be edited to have 1/sqrt(2) amplitude in both directions
        (_, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
        (forw, _) = S.GetPowerFlux(Layer = 'VacuumBelow', zOffset = 0)
        forward_power_per_wavelength[i_wavelength] = np.abs(forw)
        transmitted_power_per_wavelength[i_wavelength] = 1 - np.abs(back)
        del S

    # plt.plot(transmitted_power_per_wavelength)
    torch.save(transmitted_power_per_wavelength, f'{ff.home_directory()}/untracked-figures/grid_search/twisted-squares_{thicknesses[0]}_{widths[0]}_backward.pt')
    torch.save(forward_power_per_wavelength, f'{ff.home_directory()}/untracked-figures/grid_search/twisted-squares_{thicknesses[0]}_{widths[0]}_forward.pt')