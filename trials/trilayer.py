"""
This file simulates trials of the trilayer structure with different width, height, and thickness parameters
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
    "seeds": {
        "torch": int(52),
        "numpy": int(41)
    },
    "incidence_angle": float(0),
    "image_harmonics": int(14),
    "polarization_angle": float(45),
}

base_log_dir = os.path.join(ff.home_directory(), 'logs')
os.makedirs(base_log_dir, exist_ok=True)

log_dir = ff.get_unique_log_dir(base_log_dir, config)
config_file = os.path.join(log_dir, 'config.json')
with open(config_file, 'w') as f:
    json.dump(config, f, indent=4)

torch.manual_seed(config['seeds']['torch'])
np.random.seed(config['seeds']['numpy'])
fom_file = os.path.join(log_dir, 'fom_values.txt')

wavelengths = torch.linspace(.350, 3, 2651)  # Issue when this goes past 99?

wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)] # This is necessary due to S4 bugging out at these wavelengths

block_width_list = np.linspace(0.05, .5, 10)     # example widths in microns
block_height_list = np.linspace(0.01, 0.2, 10)   # example heights in microns
aln_thickness_list = np.linspace(0.01, 0.2, 5)   # example AlN thicknesses in microns

for block_width in block_width_list:
    for block_height in block_height_list:
        for current_aln_thickness in aln_thickness_list:
            transmitted_power_per_wavelength = torch.zeros((len(wavelengths),))
            for i_wavelength, wavelength in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
                S = S4.New(Lattice = .6, NumBasis=config['image_harmonics'])
                S.SetOptions(LanczosSmoothing=True)
                S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
                S.SetMaterial(Name='W', Epsilon=(ff.w_n[wavelengths.shape[0] // 2])**2)    # Simple approximate usage
                S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[wavelengths.shape[0] // 2])**2)
                S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vacuum')
                S.AddLayer(Name = 'Grid', Thickness = block_height, Material = 'Vacuum')
                S.SetRegionRectangle(Layer = 'Grid', Material = f'W', Center = (.3, .3), Halfwidths = (block_width, .6), Angle = 0)
                S.AddLayer(Name = 'Substrate', Thickness = current_aln_thickness, Material = 'AlN')
                S.AddLayer(Name = 'Absorber', Thickness = 1, Material = 'W')
                S.SetFrequency(1 / wavelength)
                S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2), Order=0) # In the real simulation this should be edited to have 1/sqrt(2) amplitude in both directions
                (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
                transmitted_power_per_wavelength[i_wavelength] = 1 - np.abs(back)
            transmitted_power_per_wavelength = torch.tensor(transmitted_power_per_wavelength)
            FOM = ff.power_ratio(wavelengths, transmitted_power_per_wavelength, ff.T_e, .726)
            with open(fom_file, 'a+') as f:
                msg = (
                    f"Tungsten block trial -> block_width={block_width}, "
                    f"block_height={block_height}, AlN_thickness={current_aln_thickness}\n"
                    f"FOM Value: {FOM.item()}\n{'-' * 30}\n"
                )
                f.write(msg)
                print(msg)
            emiss_file_name = (
                f"emissivity_{block_height}H_"
                f"{block_width}W_"
                f"{current_aln_thickness}AlN.txt"
            )
            emiss_file_path = os.path.join(log_dir, emiss_file_name)
            np.savetxt(emiss_file_path, transmitted_power_per_wavelength.detach().cpu().numpy())
            print(f"Emissivity profile saved to {emiss_file_path}")

            # Clean up
            del S
            gc.collect()

