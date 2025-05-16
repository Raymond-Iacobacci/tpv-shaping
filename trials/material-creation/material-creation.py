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
    np.random.seed(i)
    thickness, radius = np.random.rand(2)
    print(f'Thickness, radius: {thickness}, {radius}')
    for i_wavelength, wavelength in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
        if i_wavelength % 20 != 0:
            continue
        if wavelength == 0.5 or wavelength == 1.0:
             continue
        S = S4.New(Lattice = ((.6, 0), (0,.6)), NumBasis=config['image_harmonics'])
        S.SetOptions(LanczosSmoothing=True)
        S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
        S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)    # Simple approximate usage
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)
        S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')
        S.AddLayer(Name = 'Grid', Thickness = thickness, Material = 'AlN')

        # S.SetRegionRectangle(Layer = 'Grid', Material = f'AlN', Center = (.3, .3), Halfwidths = (np.random.rand(), .6), Angle = 0)
        if i % 2 == 0:
            S.SetRegionCircle(Layer = 'Grid', Material = f'Vacuum', Center = (.3, .3), Radius = radius * .3)
        else:
            S.SetRegionCircle(Layer = 'Grid', Material = f'W', Center = (.3, .3), Radius = radius * .3)
        S.AddLayer(Name = 'VacuumBelow', Thickness = .5, Material = 'Vacuum')


        S.SetFrequency(1 / wavelength)
        S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0) # In the real simulation this should be edited to have 1/sqrt(2) amplitude in both directions
        (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
        transmitted_power_per_wavelength[i_wavelength] = 1 - np.abs(back)
        del S

    # plt.plot(transmitted_power_per_wavelength)
    torch.save(transmitted_power_per_wavelength, f'{ff.home_directory()}/untracked-figures/grid_{thickness}_{radius}.pt')
    plt.plot((1+torch.sqrt(transmitted_power_per_wavelength))/(1-torch.sqrt(transmitted_power_per_wavelength)))
    plt.savefig(f'{ff.home_directory()}/untracked-figures/grid_{thickness}_{radius}.png');