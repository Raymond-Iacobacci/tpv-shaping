import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
import collections
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
    "noise_dimension": int(3),
    "seeds": {
        "torch": int(42),
        "numpy": int(42)
    },
    "default_gradient_scale": float(1e2),
    "learning_rate": float(7e-3),
    "binarization_scale": float(1e-11),
    "off_angle": float(45.0),
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
            filename = os.path.join(log_dir, f'{img_idx}.{args.resume_iteration-1}.npy')
            if not os.path.isfile(filename):
                raise RuntimeError(f'Cannot resume iteration={args.resume_iteration}. Missing file {filename}')
    else:
        existing_iterations = []
        for img_idx in range(1, num_images + 1):
            img_files = [
                int(fname.split('.')[1])
                for fname in os.listdir(log_dir)
                if fname.startswith(f"{img_idx}.") and fname.endswith('.npy')
            ]
            if not img_files:
                raise RuntimeError(f'No iterations found for image {img_idx}. Cannot resume.')
            existing_iterations.append(set(img_files))

        common_iterations = set.intersection(*existing_iterations)
        if not common_iterations:
            raise RuntimeError(
                "No common iterations found across all images. Cannot resume.")

        # Select the highest possible iteration
        start_iteration = max(common_iterations)+1
        print(f'Automatically selected the highest common resume_iteration: {start_iteration}')

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging = "Standard"
debug = True
save_image_per = 2


def printf(string):
    if logging:
        print(string)
        
def write_to_temp_file(line, max_lines=1000):
    temp_file = Path(ff.home_directory()) / 'transmitted_power_log.txt'
    
    # Read existing lines if file exists
    lines = []
    if temp_file.exists():
        with open(temp_file, 'r') as f:
            lines = f.readlines()
    
    # Add new line and keep only last max_lines
    lines.append(line + '\n')
    lines = lines[-max_lines:]
    
    # Write back to file
    with open(temp_file, 'w') as f:
        f.writelines(lines)

###############################################################################################


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.noise = torch.rand(
            size=(config['num_images'], config['noise_dimension']))
        self.FC = nn.Sequential(
            # nn.Linear(in_features=config['noise_dimension'],
            #           out_features=config['hidden_dimension']),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features=config['hidden_dimension'], out_features=100),
            nn.Linear(in_features = config['noise_dimension'],
            out_features=100),
        )

    def forward(self):
        output = self.FC(self.noise)
        output = nn.Sigmoid()(output)
        output = output * 1.1 - 0.05
        return torch.clamp(output, min=0.0, max=1.0)


generator = Generator()
if resuming:
    generator.load_state_dict(torch.load(
        os.path.join(log_dir, 'generator_state_dict.pth'), weights_only = True))
    generator.train()
optimizer = torch.optim.Adam(
    generator.parameters(), lr=config['learning_rate'])

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
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)][::100]
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

optimizer = torch.optim.Adam(generator.parameters())
num_cycles = 400

z_step = .1
aln_depth = .473
z_max = 3+aln_depth
z_min = -z_step

grating_z_space = torch.linspace(1 + z_step, z_max - 2 - z_step, 4)
grating_x_space = torch.linspace(-0.5, 0.49, 100) + .5  # Every 10nm

harmonics = 14

for it in range(start_iteration, num_cycles):

    avg_fom = 0

    batch_fom_wrt_perm = torch.zeros((100,))

    with open(fom_file, 'a+') as f:
        f.write(f'\nIteration {it}\n')
        f.write('-' * 30 + '\n')

    optimizer.zero_grad()
    generated_images = generator()
    read = False
    image_gradient = np.zeros((100,))
    
    current_gradient_scale = ff.read_live_gradient_scale(config['default_gradient_scale'])
    
    for image_index, image in enumerate(generated_images):
        image.requires_grad_(True)

        transmitted_power_per_wavelength = np.zeros(wavelengths.shape[0])
        angled_e_fields_per_wavelength = np.zeros(
            (wavelengths.shape[0], grating_z_space.shape[0], grating_x_space.shape[0], 3), dtype=complex)

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

            S.AddLayer(Name='Grating', Thickness=aln_depth,
                       Material='AluminumNitride')

            for q in range(0, 96):
                S.SetMaterial(Name=f'Material_{q}', Epsilon=permittivity_values[q].item())
                S.SetRegionRectangle(Layer='Grating', Material=f'Material_{q}', Center=(centers[q], 0), Halfwidths=(
                    # NOTE: if this becomes -1 this works, otherwise it doesn't...??)
                    1/200, 1 / 2), Angle=0)

            S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
            S.AddLayer(Name="AirBelow", Thickness=1, Material='Vacuum')

            S.SetExcitationPlanewave(
                IncidenceAngles=(
                    # polar angle in [0,180) -- this is the first one that we change for the angular dependence
                    config['off_angle'],
                    0  # azimuthal angle in [0,360)
                ), sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2), Order=0
            )

            S.SetOptions(
                PolarizationDecomposition=True
            )

            S.SetFrequency(1 / float(wavelength))
            (norm_forw, norm_back) = S.GetPowerFluxByOrder(Layer='TungstenBelow', zOffset=0)[0]

            transmitted_power_per_wavelength[i] = np.abs(norm_forw)

            power_line = f'{torch.round(wavelength * 1000)}nm: {transmitted_power_per_wavelength[i]}'
            write_to_temp_file(power_line)

            zc = 0
            for z in grating_z_space:
                # TODO: verify that the order of the responses matches the natural order of the x variables
                E, H = S.GetFieldsOnGrid(
                    z, NumSamples=(200, 1), Format='Array')
                angled_e_fields_per_wavelength[i][zc] = np.array(E[0])[1::2]
                zc += 1

            del S

        transmitted_power_per_wavelength = torch.tensor(
            transmitted_power_per_wavelength, requires_grad=True)
        fom = power_ratio(
            wavelengths, transmitted_power_per_wavelength, T_e, .726)
        with open(fom_file, 'a+') as f:
            f.write(f'  Image {image_index+1:2d}: {fom.item():.6f}\n')
        fom.backward()  # Loads up the gradients into the tensor by constructing the computational graph
        # volts/meter, over all wavelengths
        fom_wrt_flux = transmitted_power_per_wavelength.grad
        
        adjoint_e_fields_per_wavelength = np.zeros(
            (wavelengths.shape[0], grating_z_space.shape[0], grating_x_space.shape[0], 3), dtype=complex)
        
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

            S.AddLayer(Name='Grating', Thickness=aln_depth,
                       Material='AluminumNitride')

            for q in range(0, 96):
                S.SetMaterial(Name=f'Material_{q}', Epsilon=permittivity_values[q].item())
                S.SetRegionRectangle(Layer='Grating', Material=f'Material_{q}', Center=(centers[q], 0), Halfwidths=(
                    # NOTE: if this becomes -1 this works, otherwise it doesn't...??)
                    1/200, 1 / 2), Angle=0)

            S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
            S.AddLayer(Name="AirBelow", Thickness=1, Material='Vacuum')

            S.SetExcitationPlanewave(
                IncidenceAngles=(
                    # polar angle in [0,180) -- this is the first one that we change for the angular dependence
                    config['off_angle'],
                    0  # azimuthal angle in [0,360)
                ), sAmplitude=1, pAmplitude=1, Order=0
            ) # The scaling will be handled in the multiplication step at the end

            S.SetOptions(
                PolarizationDecomposition=True
            )

            S.SetFrequency(1 / float(wavelength))
            (forw, back) = S.GetPowerFlux(Layer='AirAbove', zOffset=0)

            # transmitted_power_per_wavelength[i] = 1-np.abs(back)

            # printf(f'{torch.round(wavelength * 1000)}nm: {transmitted_power_per_wavelength[i]}')

            zc = 0
            for z in grating_z_space:
                # TODO: verify that the order of the responses matches the natural order of the x variables
                E, H = S.GetFieldsOnGrid(
                    z, NumSamples=(200, 1), Format='Array')
                adjoint_e_fields_per_wavelength[i][zc] = np.array(E[0])[1::2]
                zc += 1

            del S
        

        fom_wrt_perm_per_wavelength_p1 = -2 * fom_wrt_flux * wavelengths ** 2 * e_0
        fom_wrt_perm_per_wavelength_p2 = np.einsum(
            'ijkl,ijkl->ijk', angled_e_fields_per_wavelength, adjoint_e_fields_per_wavelength)
        fom_wrt_perm_per_wavelength = fom_wrt_perm_per_wavelength_p1[:, np.newaxis, np.newaxis] * np.real(
            fom_wrt_perm_per_wavelength_p2)

        fom_wrt_perm = fom_wrt_perm_per_wavelength.sum(0).sum(0)
        fom_wrt_perm = fom_wrt_perm + torch.sum(.25-(.5-image)**2) * config['binarization_scale'] / config['num_images']
        fom_wrt_perm = fom_wrt_perm * current_gradient_scale
        print(f'Current gradient scale: {current_gradient_scale}')
        print(f'Gradient magnitude: {torch.mean(torch.abs(fom_wrt_perm))}')
        print(f'Gradient to (binarization) lambda ratio: {np.mean(np.abs(fom_wrt_perm.detach().numpy())) / (4 * config["binarization_scale"] / config["num_images"])}')
        print(f'Binarization: {torch.mean(2*torch.abs(image - 0.5))*100:.2f}%')
        # Averaging over all wavelengths (already accounted for the the adjoint step) and averaging over all z-points
        batch_fom_wrt_perm += fom_wrt_perm
        print(f"Gradient: {batch_fom_wrt_perm.grad}")
        print(f"Magnitude: {batch_fom_wrt_perm}")
        avg_fom += fom.detach().cpu().numpy()

    generated_images.backward(
        -batch_fom_wrt_perm.repeat(config['num_images'], 1))
    optimizer.step()
    
    torch.save(generator.state_dict(), os.path.join(log_dir, 'generator_state_dict.pth'))
    
    avg_fom /= config['num_images']
    with open(fom_file, 'a+') as f:
        f.write(f'Average FOM: {avg_fom}\n')

    for image_index, image in enumerate(generated_images):
        if it % save_image_per == 0:
            image_filename = os.path.join(
                log_dir, f"{image_index + 1}.{it}.npy")
            np.save(image_filename,
                    arr=generated_images[image_index].detach().cpu().numpy())
            printf(f'Saved image to {image_filename}')
