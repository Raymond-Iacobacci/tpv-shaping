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
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)][::50]


def IQE(wavelength, e_g):
    lambda_g = np.ceil(1240 / e_g) / 1000.0

    if (lambda_g > wavelength[-1]):
        l_index = wavelength[-1]
    else:
        l_index = torch.where(wavelength >= lambda_g)[0][0]
    IQE = torch.ones(len(wavelength))
    for i in range(l_index, len(wavelength)):
        IQE[i] = 0
    return IQE


def JV(em, IQE, lambda_i):
    em = em.squeeze()
    J_L = ff.q * torch.sum(em * ff.nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = ff.q * torch.sum(ff.nb_B_PV*IQE) * (lambda_i[1] - lambda_i[0])

    V_oc = (ff.k_B*ff.T_PV/ff.q)*torch.log(J_L/J_0+1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc

    J = J_L-J_0*(torch.exp(ff.q*V/(ff.k_B*ff.T_PV))-1)
    P = V*J

    return torch.max(P)


def power_ratio(lambda_i, emissivity_dataset, T_emitter, E_g_PV):
    emissivity = emissivity_dataset.squeeze()
    P_emit = torch.sum(emissivity*ff.Blackbody(lambda_i, T_emitter)
                       ) * (lambda_i[1] - lambda_i[0])
    IQE_PV = IQE(lambda_i, E_g_PV)
    JV_PV = JV(emissivity, IQE_PV, lambda_i)

    FOM = JV_PV / P_emit
    return FOM


base_config = {
    "seeds": {
        "torch": int(42),
        "numpy": int(42)
    },
    "off_angle": float(0.0),
    "square_x_location": float(0.15),
    "square_y_location": float(0.15),
    "polarization_angle": float(45),
    "square_half_widths": float(0.15),
    "wavelength_start_index": int(0),  # default0
    "wavelength_end_index": int(2649),  # default2649
    "measurement_location": int(0),
    "harmonics": int(400),
}

save = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################################


n_all = np.load('/home/rliacobacci/Downloads/n_allHTMats.npz')
k_all = np.load('/home/rliacobacci/Downloads/k_allHTMats.npz')


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
    # Load the original refractive index data.
    orig_wl, orig_n = load_refractive_index_file(filepath)
    # Assume the loaded wavelengths (orig_wl) are in micrometers.

    # Create new wavelengths from 350 nm to 3000 nm.
    new_wl_nm = np.arange(350, 3001)  # 350, 351, ..., 3000 nm
    # Convert the new wavelengths to micrometers for interpolation.
    new_wl_microns = new_wl_nm / 1000.0

    # Create the cubic spline interpolation function.
    cs = CubicSpline(orig_wl, orig_n)
    # Get the interpolated refractive indices.
    new_n = cs(new_wl_microns)

    # Construct a dictionary mapping wavelength (in nm) to refractive index.
    refractive_indices = {wl: n_val for wl, n_val in zip(new_wl_nm, new_n)}
    return refractive_indices


SiO2_n = interpolate_refractive_indices(
    '/home/rliacobacci/Downloads/SiO2_n.txt')

w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j

harmonic_values = [400]
for harmonics in harmonic_values:
    print(f"\n=== Running simulation with harmonics = {harmonics} ===\n")
    
    # Create a copy of the base config and update the harmonics value
    config = base_config.copy()
    config["harmonics"] = int(harmonics)

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
        S.AddLayer(Name='TungstenGrid', Thickness=.06, Material='Vacuum')
        S.SetRegionRectangle(Layer='TungstenGrid', Material='Tungsten', Center=(
            config['square_x_location'] + config['square_half_widths'], config['square_y_location'] + config['square_half_widths']), Halfwidths=(config['square_half_widths'], config['square_half_widths']), Angle=0)
        S.AddLayer(Name='SiliconDioxide', Thickness=.06, Material='SiliconDioxide')
        S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
        S.AddLayer(Name="AirBelow", Thickness=1, Material='Vacuum')

        S.SetExcitationPlanewave(
            IncidenceAngles=(
                # polar angle in [0,180) -- this is the first one that we change for the angular dependence
                config['off_angle'],
                0  # azimuthal angle in [0,360)
            ), sAmplitude=np.cos(config['polarization_angle']*np.pi/180), pAmplitude=np.sin(config['polarization_angle']*np.pi/180), Order=0
        )
        S.SetOptions(
            PolarizationDecomposition=True
        )

        S.SetFrequency(1 / float(wavelength))
        # (_, back1) = S.GetPowerFlux(Layer='TungstenGrid', zOffset=0.06)
        # (forw1, _) = S.GetPowerFlux(Layer='TungstenBelow', zOffset=0)
        (_, back) = S.GetPowerFlux(Layer='AirAbove', zOffset=0.5)
        # (_, back2) = S.GetPowerFlux(Layer='SiliconDioxide', zOffset=0)
        # (forw2, _) = S.GetPowerFlux(Layer='SiliconDioxide', zOffset=0.06)
        # (_, back3) = S.GetPowerFlux(Layer='TungstenGrid', zOffset=0)
        # (_, back4) = S.GetPowerFlux(Layer='AirAbove', zOffset=1)
        # print(np.abs(forw) - np.abs(back))
        # orders = S.GetBasisSet()
        total_exit_flux = 0
        # for order in S.GetPowerFluxByOrder(Layer='AirAbove', zOffset=.5):
            # Get the power flux for each diffraction order
            # total_exit_flux += np.abs(order[1])

        if config['measurement_location'] == 0:
            transmitted_power_per_wavelength.append(1 - np.abs(back))
        # elif config['measurement_location'] == 1:
        #     transmitted_power_per_wavelength.append(np.abs(forw1))
        # elif config['measurement_location'] == 2:
        #     transmitted_power_per_wavelength.append(1-(np.abs(back1)))
        # elif config['measurement_location'] == 3:
        #     transmitted_power_per_wavelength.append(1-np.abs(back2))
        # elif config['measurement_location'] == 4:
        #     transmitted_power_per_wavelength.append(np.abs(forw2))
        # elif config['measurement_location'] == 5:
        #     transmitted_power_per_wavelength.append(
        #         1-np.abs(back)+(np.abs(back3) - np.abs(back1)))
        # elif config['measurement_location'] == 6:
        #     transmitted_power_per_wavelength.append(1 - np.abs(back4))
        # elif config['measurement_location'] == 7:
        #     transmitted_power_per_wavelength.append(1-total_exit_flux)

        print(
            f'{torch.round(wavelength * 1000)}nm: {transmitted_power_per_wavelength[-1]}')
        

        del S

    if save:

        config_str = json.dumps(config, sort_keys=True)
        # Create a hash of the config for unique identification
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        base_name = f"harmonic-logs/rotated-single-dim-config_{config_hash}"
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
        # plt.show()
