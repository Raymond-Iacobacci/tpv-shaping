'''
This file repeats random image calculations until it gets one that performs better than the homogeneous case.
'''

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
import torch.nn as nn

config = {
    "seeds": {
        "torch": int(42),
        "numpy": int(42)
    },
    "off_angle": float(30.0),
    "homogeneous": False,
}

torch.manual_seed(config['seeds']['torch'])
np.random.seed(config['seeds']['numpy'])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################################

h = 6.626070e-34  # Js Planck's constant
c = 2.997925e8  # m/s speed of light
k_B = 1.380649e-23  # J/K Boltzmann constant
q = 1.602176e-19  # C elementary charge
e_0 = 8.8541878128e-12


def Blackbody(lambda_i, T):
    return (2*h*c**2) / ((np.exp((h*c)/(k_B*T*lambda_i*1e-6))-1)*lambda_i**5)*1e14


def nb_B(lambda_i, T):
    return (2*c) / ((np.exp((h*c)/(k_B*T*lambda_i*1e-6))-1)*lambda_i**4)*1e8


wavelengths = torch.linspace(.350, 3, 2651)  # Issue when this goes past 99?
# [::100] # This is necessary due to S4 bugging out at these wavelengths
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)]
T_e = 2073.15  # K emitter temperature
nb_B_e = nb_B(wavelengths, T_e)  # 2073.15K photon
T_PV = 300  # K PV temperature
nb_B_PV = nb_B(wavelengths, T_PV)  # 300K photon


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
    J_L = q * torch.sum(em * nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = q * torch.sum(nb_B_PV*IQE) * (lambda_i[1] - lambda_i[0])

    V_oc = (k_B*T_PV/q)*torch.log(J_L/J_0+1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc

    J = J_L-J_0*(torch.exp(q*V/(k_B*T_PV))-1)
    P = V*J

    return torch.max(P)


def power_ratio(lambda_i, emissivity_dataset, T_emitter, E_g_PV):
    emissivity = emissivity_dataset.squeeze()
    P_emit = torch.sum(emissivity*Blackbody(lambda_i, T_emitter)
                       ) * (lambda_i[1] - lambda_i[0])
    IQE_PV = IQE(lambda_i, E_g_PV)
    JV_PV = JV(emissivity, IQE_PV, lambda_i)

    FOM = JV_PV / P_emit
    return FOM


n_all = np.load('/home/rliacobacci/Downloads/n_allHTMats.npz')
k_all = np.load('/home/rliacobacci/Downloads/k_allHTMats.npz')

w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j

z_step = .1
aln_depth = .473
z_max = 3+aln_depth
z_min = -z_step

grating_z_space = torch.linspace(1 + z_step, z_max - 2 - z_step, 4)
grating_x_space = torch.linspace(-0.5, 0.49, 100) + .5  # Every 10nm

harmonics = 14

batch_fom_wrt_perm = torch.zeros((100,))

read = False

transmitted_power_per_wavelength = np.zeros(wavelengths.shape[0])
angled_e_fields_per_wavelength = np.zeros(
    (wavelengths.shape[0], grating_z_space.shape[0], grating_x_space.shape[0], 3), dtype=complex)

config['homogeneous'] = True


def calc_fom(image):
    for i, wavelength in enumerate(wavelengths):

        # TODO: change to 14 when actually using the grating
        S = S4.New(Lattice=((1, 0), (0, 1)), NumBasis=harmonics)

        S.SetMaterial(Name='AluminumNitride', Epsilon=(aln_n[i]) ** 2)
        S.SetMaterial(Name='Tungsten', Epsilon=(w_n[i])**2)
        S.SetMaterial(Name='Vacuum', Epsilon=(1 + 0j)**2)

        S.AddLayer(Name='AirAbove', Thickness=1, Material='Vacuum')
        permittivity_values = image * (aln_n[i]**2 - 1) + 1

        width = (.5 - -.5) / len(permittivity_values)

        n_squares = len(permittivity_values)
        centers = torch.linspace(-.5 + width /
                                    2, .5 - width / 2, n_squares) + .5

        if not config["homogeneous"]:
            S.AddLayer(Name='Grating', Thickness=aln_depth,
                        Material='AluminumNitride')
            
            for q in range(0, 96):
                S.SetMaterial(Name=f'Material_{q}', Epsilon=permittivity_values[q].item())
                S.SetRegionRectangle(Layer='Grating', Material=f'Material_{q}', Center=(centers[q], 0), Halfwidths=(
                    # NOTE: if this becomes -1 this works, otherwise it doesn't...??)
                    1/200, 1 / 2), Angle=0)
        else:

            S.AddLayer(Name = 'AlN', Thickness=aln_depth, Material='AluminumNitride')
        S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
        S.AddLayer(Name="AirBelow", Thickness=1, Material='Vacuum')

        S.SetExcitationPlanewave(
            IncidenceAngles=(
                # polar angle in [0,180) -- this is the first one that we change for the angular dependence
                config['off_angle'],
                0  # azimuthal angle in [0,360)
            ), sAmplitude=0, pAmplitude=1, Order=0
        )

        S.SetOptions(
            PolarizationDecomposition=True
        )

        S.SetFrequency(1 / float(wavelength))
        (nforw, nback) = S.GetPowerFluxByOrder(Layer='TungstenBelow', zOffset=0)[0]

        transmitted_power_per_wavelength[i] = np.abs(nforw)
        # print(f'{torch.round(wavelength * 1000)}nm: {transmitted_power_per_wavelength[i]}')

        # zc = 0
        # for z in grating_z_space:
        #     # TODO: verify that the order of the responses matches the natural order of the x variables
        #     E, H = S.GetFieldsOnGrid(
        #         z, NumSamples=(200, 1), Format='Array')
        #     angled_e_fields_per_wavelength[i][zc] = np.array(E[0])[1::2]
        #     zc += 1

        del S
    
    image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
    base_dir = Path(os.path.expanduser('~')) / 'tpv_backup'
    hash_dir = base_dir / image_hash
    hash_dir.mkdir(parents=True, exist_ok=True)
    config_path = hash_dir / 'config.json'
    spectrum_path = hash_dir / 'spectrum.png'
    emissivity_path = hash_dir / 'emissivity.npy'
    
    np.save(emissivity_path, transmitted_power_per_wavelength)
    print(f'Saved emissivity profile to: {emissivity_path}')

    plt.plot(wavelengths, transmitted_power_per_wavelength)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmitted Power')
    plt.title('Transmitted Power vs. Wavelength')
    plt.savefig(spectrum_path)
    print(f'Saved spectrum plot to: {spectrum_path}')
    # plt.show()

    fom = power_ratio(wavelengths, torch.from_numpy(transmitted_power_per_wavelength), T_e, .726)
    print(f'FOM: {fom}')

    # Save config and image data
    full_config = {
        **config,  # Unpack existing config
        'image': image.tolist(),  # Add image data
        'fom': fom.item(),  # Add FOM
    }
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    return fom

baseline_fom = calc_fom(np.zeros(100))
config['homogeneous'] = False
current_seed = config['seeds']['numpy']
attempts = 0
while attempts < 100:
    attempts += 1
    config['seeds']['numpy'] = current_seed + attempts
    np.random.seed(config['seeds']['numpy'])
    image = np.random.rand(100,)
    current_fom = calc_fom(image)
    if current_fom > baseline_fom:
        print(f"Success: {current_fom}, {config['seeds']['numpy']}!")
        sys.exit(1)
    print(f"Failed: {current_fom}, {config['seeds']['numpy']}!")