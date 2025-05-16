import argparse
import csv
import hashlib
import itertools
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
from scipy.interpolate import CubicSpline

wavelengths = torch.linspace(.350, 3, 2651)  # Issue when this goes past 99?
# [::100] # This is necessary due to S4 bugging out at these wavelengths
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)]

nb_B_e = ff.nb_B(wavelengths, ff.T_e)  # 2073.15K photon
nb_B_PV = ff.nb_B(wavelengths, ff.T_PV)  # 300K photon

config = {
    "seeds": {
        "torch": int(42),
        "numpy": int(42)
    },
    "off_angle": float(0.0),
    "square_x_location": float(0.15),
    "square_y_location": float(0.15),
    "polarization_angle": float(45),
    "rectangle_x_half_width": float(0.15),
    "rectangle_y_half_width": float(0.15),
    "wavelength_start_index": int(0),  # default0
    "wavelength_end_index": int(2649),  # default2649
    # "harmonics": int(400),
    "harmonics": int(5),
    "material": str("AluminumNitride"),
}

save = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################################


n_all = np.load('../data/n_allHTMats.npz')
k_all = np.load('../data/k_allHTMats.npz')


def load_refractive_index_file(filepath):
    """
    Load a file with two columns (wl and n) and return arrays of wavelengths and refractive indices.
    Expects the file to have a header line like: "wl	n".
    """
    wavelengths = []
    n_values = []
    with open(filepath, "r") as f:
        header = f.readline()  # Skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # splits on whitespace
            if len(parts) != 2:
                continue  # or you can raise an error
            wl, n_val = parts
            wavelengths.append(float(wl))
            n_values.append(float(n_val))
    return np.array(wavelengths), np.array(n_values)


def interpolate_refractive_indices(filepath):
    orig_wl, orig_n = load_refractive_index_file(filepath)
    new_wl_nm = np.arange(350, 3001)  # 350, 351, ..., 3000 nm
    new_wl_microns = new_wl_nm / 1000.0
    cs = CubicSpline(orig_wl, orig_n)
    new_n = cs(new_wl_microns)
    refractive_indices = {wl: n_val for wl, n_val in zip(new_wl_nm, new_n)}
    return refractive_indices


SiO2_n = interpolate_refractive_indices(
    '../data/SiO2_n.txt')

w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j


transmitted_power_per_wavelength = []

for i, wavelength in enumerate(wavelengths[config['wavelength_start_index']:config['wavelength_end_index']]):
    i = i + config['wavelength_start_index']
    # TODO: change to 14 when actually using the grating
    S = S4.New(Lattice=((.6, 0), (0, .6)), NumBasis=config['harmonics'])

    S.SetMaterial(Name='AluminumNitride', Epsilon=(aln_n[i]) ** 2)
    S.SetMaterial(Name='Tungsten', Epsilon=(w_n[i])**2)
    S.SetMaterial(Name='Vacuum', Epsilon=(1 + 0j)**2)
    S.SetMaterial(Name='SiliconDioxide', Epsilon=(
        SiO2_n[int(wavelength*1e3)]**2))

    S.AddLayer(Name='AirAbove', Thickness=1, Material='Vacuum')
    # S.AddLayer(Name='TungstenGrid', Thickness=.06, Material='Vacuum')
    # S.SetRegionRectangle(Layer='TungstenGrid', Material='Tungsten', Center=(config['square_x_location'] + config['rectangle_x_half_width'], config['square_y_location'] + config['rectangle_y_half_width']), Halfwidths=(config['rectangle_x_half_width'], config['rectangle_y_half_width']), Angle=0)
    S.AddLayer(Name='SiliconDioxide', Thickness=.06, Material=config['material'])
    S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
    S.AddLayer(Name="AirBelow", Thickness=1, Material='Vacuum')

    S.SetExcitationPlanewave(
        IncidenceAngles=(config['off_angle'], 0), 
        sAmplitude=np.cos(config['polarization_angle']*np.pi/180),
        pAmplitude=np.sin(config['polarization_angle']*np.pi/180),
        Order=0
    )
    S.SetOptions(PolarizationDecomposition=True)

    S.SetFrequency(1 / float(wavelength))
    (_, back) = S.GetPowerFlux(Layer='AirAbove', zOffset=0.5)
    transmitted_power_per_wavelength.append(1 - np.abs(back))

    print(f'{torch.round(wavelength * 1000)}nm: {transmitted_power_per_wavelength[-1]}')
    
    del S
transmitted_power_per_wavelength = torch.tensor(transmitted_power_per_wavelength)
# print("Power:",ff.power_ratio(wavelengths, transmitted_power_per_wavelength, ff.T_e, .726))
if save:

    config_str = json.dumps(config, sort_keys=True)
    # Create a hash of the config for unique identification
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    base_name = f'logs/zhao_{config_hash}'
    os.makedirs(base_name, exist_ok=True)
    config_file = os.path.join(base_name, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

    np.save(f"{base_name}/emission", transmitted_power_per_wavelength)

    plt.plot(wavelengths[config['wavelength_start_index']
             :config['wavelength_end_index']], transmitted_power_per_wavelength)
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Transmitted Power')
    plt.title('Transmitted Power vs. Wavelength')
    plt.savefig(f"{base_name}/spectrum.png")
    plt.show()
