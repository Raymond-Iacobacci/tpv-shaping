"""
This file optimizes images with the explicit usage of a generator. It's currently being updated for usage with Simulation.GetFieldsOnGrid instead of Simulation.GetFields, in order to speed up the process significantly.
"""


import argparse
import collections
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
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--resume-iteration', type=int, default=None,
                    help="Iteration to resume from")
parser.add_argument('--resume-choice', type=str, default=None,
                    help="If multiple directories match, specify which one to resume")

args = parser.parse_args()

config = {
    "num_images": int(1),
    "hidden_dimension": int(10),
    "noise_dimension": int(8),
    "seeds": {
        "torch": int(42),
        "numpy": int(42)
    },
    "default_gradient_scale": float(1e2),
    "learning_rate": float(7e-3),
    "binarization_scale": float(1e-11),
    "off_angle": float(0),
}

base_log_dir = os.path.join(ff.home_directory(), 'logs')
os.makedirs(base_log_dir, exist_ok=True)

if (args.resume_iteration is not None) and (args.resume_choice is None):
    resume_choice = input(
        "Multiple directories match the given iteration. Please specify which one to resume: ")
    args.resume_choice = resume_choice
resuming = args.resume_choice is not None

if resuming:  # Must choose iteration from at most one more than iterations already completed
    candidate_dirs = os.scandir(base_log_dir)
    if not candidate_dirs:
        raise RuntimeError(
            "No directories match the given seeds & scale. Cannot resume.")
    chosen = None
    for d in candidate_dirs:
        if os.path.basename(d) == args.resume_choice:
            chosen = d
            break
    if not chosen:
        raise RuntimeError("No directories match the name. Cannot resume.")
    log_dir = chosen
    print(f'Resuming from {log_dir}')

    resume_config_path = os.path.join(log_dir, 'config.json')
    with open(resume_config_path, 'r') as f:
        resume_config = json.load(f)
    config = resume_config
    num_images = config['num_images']
    if args.resume_iteration is not None:
        for img_idx in range(1, num_images + 1):
            filename = os.path.join(
                log_dir, f'{img_idx}.{args.resume_iteration-1}.npy')
            if not os.path.isfile(filename):
                raise RuntimeError(
                    f'Cannot resume iteration={args.resume_iteration}. Missing file {filename}')
    else:
        existing_iterations = []
        for img_idx in range(1, num_images + 1):
            img_files = [
                int(fname.split('.')[1])
                for fname in os.listdir(log_dir)
                if fname.startswith(f"{img_idx}.") and fname.endswith('.npy')
            ]
            if not img_files:
                raise RuntimeError(
                    f'No iterations found for image {img_idx}. Cannot resume.')
            existing_iterations.append(set(img_files))

        common_iterations = set.intersection(*existing_iterations)
        if not common_iterations:
            raise RuntimeError(
                "No common iterations found across all images. Cannot resume.")

        # Select the highest possible iteration
        start_iteration = max(common_iterations)+1
        print(
            f'Automatically selected the highest common resume_iteration: {start_iteration}')

else:

    def get_unique_log_dir(base_dir, config):
        """
        Generates a unique directory name based on the config.
        If a directory with the same config exists, appends a suffix.
        """
        # Serialize the config to a JSON string
        config_str = json.dumps(config, sort_keys=True)
        # Create a hash of the config for unique identification
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        base_name = f"config_{config_hash}"
        log_dir = os.path.join(base_dir, base_name)

        # If directory exists, append a suffix
        suffix = 1
        while os.path.exists(log_dir):
            log_dir = os.path.join(base_dir, f"{base_name}_{suffix}")
            suffix += 1

        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    log_dir = get_unique_log_dir(base_log_dir, config)

    config_file = os.path.join(log_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

    start_iteration = 0

torch.manual_seed(config['seeds']['torch'])
np.random.seed(config['seeds']['numpy'])
fom_file = os.path.join(log_dir, 'fom_values.txt')

wavelengths = torch.linspace(.350, 3, 2651)  # Issue when this goes past 99?

wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)] # This is necessary due to S4 bugging out at these wavelengths
T_e = 2073.15  # K emitter temperature
nb_B_e = ff.nb_B(wavelengths, T_e)  # 2073.15K photon
T_PV = 300  # K PV temperature
nb_B_PV = ff.nb_B(wavelengths, T_PV)  # 300K photon

n_all = np.load('/home/rliacobacci/Downloads/n_allHTMats.npz')
k_all = np.load('/home/rliacobacci/Downloads/k_allHTMats.npz')

w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j

num_cycles = 400

vacuum_thickness = .5 # microns
grid_thickness = .473
absorb_thickness = 1

y_pt = 0
z_dots, y_dots, x_dots, x_msrmt_dots, num_image_squares = 100, 1, 100, 50, 50
z_space = np.linspace(0, vacuum_thickness+grid_thickness+absorb_thickness, z_dots)
z_grtg_space = z_space[(z_space >= vacuum_thickness) & (z_space <= vacuum_thickness + grid_thickness)]
z_buffer = .15 # This is for when we offset the measurement later on
y_space = np.linspace(y_pt, y_pt, y_dots)
x_space = np.linspace(0, 1, x_dots) # Used for plotting
x_msrmt_space = np.array([(q+1)/x_msrmt_dots - 1/(2*x_msrmt_dots) for q in range(x_msrmt_dots)])
x_grtg_space = np.array([(q+1) / num_image_squares - 1 / (2 * num_image_squares) for q in range(num_image_squares)])

n_harmonics = 25
excite_harmonics = 100


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
    J_L = ff.q * torch.sum(em * nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = ff.q * torch.sum(nb_B_PV*IQE) * (lambda_i[1] - lambda_i[0])

    V_oc = (ff.k_B*T_PV/ff.q)*torch.log(J_L/J_0+1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc

    J = J_L-J_0*(torch.exp(ff.q*V/(ff.k_B*T_PV))-1)
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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.noise = torch.rand(
            size=(config['num_images'], config['noise_dimension']))
        self.FC = nn.Sequential(
            nn.Linear(in_features=config['noise_dimension'],
                      out_features=config['hidden_dimension']),
            nn.ReLU(),
            nn.Linear(
                in_features=config['hidden_dimension'], out_features=num_image_squares),
            # nn.Linear(in_features=config['noise_dimension'],
                    #   out_features=num_image_squares),
        )
        # Uncomment the following lines in order to force the first round of generated images to near 1 everywhere.
        # with torch.no_grad():
        #     # Set weights to small values to reduce input variation
        #     self.FC[0].weight.fill_(0.1)
        #     # Set biases to high values to push sigmoid outputs close to 1
        #     # A bias of 10 gives sigmoid(10) ≈ 0.9999
        #     self.FC[0].bias.fill_(10.0)

    def forward(self):
        output = self.FC(self.noise)
        output = nn.Sigmoid()(output)
        output = output * 1.1 - 0.05
        return torch.clamp(output, min=0.0, max=1.0)


generator = Generator()
if resuming:
    generator.load_state_dict(torch.load(
        os.path.join(log_dir, 'generator_state_dict.pth'), weights_only=True))
    generator.train()
optimizer = torch.optim.Adam(
    generator.parameters(), lr=config['learning_rate'])

for it in range(start_iteration, num_cycles):
    avg_fom = 0

    with open(fom_file, 'a+') as f:
        f.write(f'\nIteration {it}\n')
        f.write('-' * 30 + '\n')

    optimizer.zero_grad()
    generated_images = generator()
    dfom_deps = torch.zeros((num_image_squares,))
    
    for image_index, image in enumerate(generated_images):
        image.requires_grad_(True)
        
        # Initialize wavelength-specific variables:
        transmitted_power_per_wavelength = torch.zeros((len(wavelengths),))
        dflux_deps_all_wavelength = np.zeros((wavelengths.shape[0], num_image_squares))
        
        for i_wavelength, wavelength in enumerate(wavelengths):
            itr_start = time.time()
    
            S = S4.New(Lattice = 1, NumBasis=n_harmonics) # This sets nonzero harmonics only in a single (x) direction, so we actually have 14 harmonics in that direction
            S.SetOptions(LanczosSmoothing=True)

            S.SetMaterial(Name='W', Epsilon=(w_n[i_wavelength])**2)
            S.SetMaterial(Name='Vacuum', Epsilon=(1 + 0j)**2)
            S.SetMaterial(Name='AlN', Epsilon=(aln_n[i_wavelength])**2)

            S.AddLayer(Name = 'VacuumAbove', Thickness = vacuum_thickness, Material = 'Vacuum')
            S.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'AlN')
            # S.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'Vacuum')

            for q in range(num_image_squares):
                # image[q].item()
                S.SetMaterial(Name = f'Grid material {q}', Epsilon = image[q].item() * (aln_n[i_wavelength]**2 - 1) + 1)
                S.SetRegionRectangle(Layer = 'Grid', Material = f'Grid material {q}', Center = ((q+1)/num_image_squares - 1/(2*num_image_squares), .5), Halfwidths = (1/(2*num_image_squares), 1/2), Angle = 0)

            S.AddLayer(Name = 'Absorber', Thickness = absorb_thickness, Material = 'W')

            S.SetExcitationPlanewave(IncidenceAngles=(config['off_angle'], 0), sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2), Order=0) # In the real simulation this should be edited to have 1/sqrt(2) amplitude in both directions

            S.SetFrequency(1 / wavelength)
            
            (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
            
            transmitted_power_per_wavelength[i_wavelength] = 1 - np.abs(back)

            msr_start = time.time()
            
            # points = np.zeros((x_msrmt_dots, 3))
            # points[:, 0] = x_msrmt_space
            # points[:, 1] = y_pt
            # points[:, 2] = z_buffer
            
            # E, H = S.GetFieldsOnGrid(points)
            
            adj_src_fields = np.zeros((1, y_dots, x_msrmt_dots), dtype = complex)
            # adj_src_fields[0, y_dots, :] = E[:, 0]
            
            for ix, x in enumerate(x_msrmt_space):
                adj_src_fields[0, 0, ix] = S.GetFields(x, y_pt, z_buffer)[0][0] # This computes the electric field. The .25 buffer is to handle edge effects. To obtain the z-flux of the electric field, we need to store the first element of the last dimension via the RHR.
            grtg_fields = np.zeros((z_grtg_space.shape[0], y_dots, x_grtg_space.shape[0], 3), dtype = complex)
            for iz, z in enumerate(z_grtg_space):
                for ix, x in enumerate(x_grtg_space):
                    grtg_fields[iz, y_pt, ix] = S.GetFields(x, y_pt, z)[0]
                    
            msr_end = time.time()
            del S
            
            S2 = S4.New(Lattice = 1, NumBasis = n_harmonics)
            S2.SetOptions(Verbosity = 0)

            S2.SetMaterial(Name = 'Vacuum', Epsilon = (1 + 0j)**2)
            S2.SetMaterial(Name = 'W', Epsilon = (w_n[i_wavelength])**2)
            S2.AddLayer(Name = 'VacuumAbove', Thickness = vacuum_thickness - z_buffer, Material = 'Vacuum')
            # S2.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'AlN')
            S2.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'Vacuum')

            for q in range(num_image_squares):
                # image[q].item()
                S2.SetMaterial(Name = f'Grid material {q}', Epsilon = image[q].item() * (aln_n[i_wavelength]**2 - 1) + 1)
                S2.SetRegionRectangle(Layer = 'Grid', Material = f'Grid material {q}', Center = ((q+1)/num_image_squares - 1/(2*num_image_squares), .5), Halfwidths = (1/(2*num_image_squares), 1/2), Angle = 0)

            S2.AddLayer(Name = 'Absorber', Thickness = absorb_thickness, Material = 'W')

            cartesian_excitation_magnitude = np.mean(np.abs(adj_src_fields[0, 0, :]))
            cartesian_excitation_phase = np.angle(adj_src_fields[0, 0, :])
            cartesian_excitations = cartesian_excitation_magnitude * np.exp(1j * cartesian_excitation_phase)
            
            excitations = ff.create_step_excitation(basis = S2.GetBasisSet(), step_values = cartesian_excitations, num_harmonics = excite_harmonics, x_shift = 0, initial_phase = 0, amplitude = 1.0, plot_fourier=False)
            
            S2.SetExcitationExterior(Excitations = tuple(excitations))
            S2.SetFrequency(1 / wavelength)

            msr2_start = time.time()

            adj_grtg_fields = np.zeros((z_grtg_space.shape[0], y_dots, x_grtg_space.shape[0], 3), dtype = complex)
            for iz, z in enumerate(z_grtg_space):
                z -= z_buffer
                for ix, x in enumerate(x_grtg_space):
                    adj_grtg_fields[iz, y_pt, ix] = S2.GetFields(x, y_pt, z)[0]

            msr2_end = time.time()
            
            del S2
            
            # Gradient calculations WRT FOM
            dflux_deps = 2 * wavelength ** 2 * ff.e_0 * np.real(np.einsum('ijkl,ijkl->ijk', grtg_fields, adj_grtg_fields)) # The negative sign and the 2 in the adjoint source construction can factor out of the real function and the einsum function
            dflux_deps_all_wavelength[i_wavelength] = torch.mean(dflux_deps, dim = 0).squeeze() # This removes the z-dimension and the collapsed y-dimension from this simulation, giving us a gradient of dimension 1 / num_image_squares
            itr_end = time.time()
            print(f'Percent measuring: {(msr_end - msr_start + msr2_end - msr2_start) / (itr_end - itr_start) * 100:.2f}%')
            
        dflux_deps_all_wavelength = torch.tensor(dflux_deps_all_wavelength, requires_grad = True)

        # Need to add scaling binary constraint here
        transmitted_power_per_wavelength = torch.tensor(transmitted_power_per_wavelength, requires_grad=True)
        fom = power_ratio(wavelengths, transmitted_power_per_wavelength, ff.T_e, .726)
        
        fom.backward()

        dfom_dflux = transmitted_power_per_wavelength.grad
        dfom_deps += torch.mean(dfom_dflux.unsqueeze(1).expand(wavelengths.shape[0], num_image_squares) * dflux_deps_all_wavelength, dim = 0) * 1e6
        print(fom, torch.mean(torch.abs(dfom_deps)))
        print(generated_images.numpy())

    generated_images.backward(
        dfom_deps.repeat(config['num_images'], 1)/config['num_images'] # Should we add a negative here, like in the original material?
    )
    optimizer.step()
    
    # torch.save(generator.state_dict(), os.path.join(log_dir, 'generator_state_dict.pth'))

    