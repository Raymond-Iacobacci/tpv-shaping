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

# -----------------------
# Command-line arguments
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume-iteration', type=int, default=None,
                    help="Iteration to resume from")
parser.add_argument('--resume-choice', type=str, default=None,
                    help="If multiple directories match, specify which one to resume")
args = parser.parse_args()

# -----------------------
# Configuration dictionary
# -----------------------
config = {
    "num_images": 1,  # Not used now, but kept for logging consistency
    "hidden_dimension": 10,
    "noise_dimension": 3,
    "seeds": {
        "torch": 45,
        "numpy": 45
    },
    "default_gradient_scale": 1e2,
    "learning_rate": 7e-3,
    "binarization_scale": 1e-11,
    "off_angle": 50.0,
}

# -----------------------
# Setup logging directory
# -----------------------
base_log_dir = os.path.join(ff.home_directory(), 'logs')
os.makedirs(base_log_dir, exist_ok=True)

if (args.resume_iteration is not None) and (args.resume_choice is None):
    resume_choice = input("Multiple directories match the given iteration. Please specify which one to resume: ")
    args.resume_choice = resume_choice
resuming = args.resume_choice is not None

if resuming:
    candidate_dirs = os.scandir(base_log_dir)
    chosen = None
    for d in candidate_dirs:
        if os.path.basename(d) == args.resume_choice:
            chosen = d
            break
    if not chosen:
        raise RuntimeError("No directories match the name. Cannot resume.")
    log_dir = chosen.path
    print(f'Resuming from {log_dir}')
    # For a random image generator, we simply load the config.
    resume_config_path = os.path.join(log_dir, 'config.json')
    with open(resume_config_path, 'r') as f:
        resume_config = json.load(f)
    config = resume_config
    start_iteration = args.resume_iteration if args.resume_iteration is not None else 0
else:
    def get_unique_log_dir(base_dir, config):
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        base_name = f"config_{config_hash}"
        log_dir = os.path.join(base_dir, base_name)
        suffix = 1
        while os.path.exists(log_dir):
            log_dir = os.path.join(base_dir, f"{base_name}_{suffix}")
            suffix += 1
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    log_dir = get_unique_log_dir(base_log_dir, config)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    start_iteration = 0

# -----------------------
# Set random seeds
# -----------------------
torch.manual_seed(config['seeds']['torch'])
np.random.seed(config['seeds']['numpy'])

# -----------------------
# File to log FOM values
# -----------------------
fom_file = os.path.join(log_dir, 'fom_values.txt')

# -----------------------
# Physical constants and helper functions
# -----------------------
h = 6.626070e-34     # Planck's constant [J.s]
c = 2.997925e8      # Speed of light [m/s]
k_B = 1.380649e-23  # Boltzmann constant [J/K]
q = 1.602176e-19    # Elementary charge [C]
e_0 = 8.8541878128e-12

def Blackbody(lambda_i, T):
    # lambda_i in micrometers
    return (2 * h * c**2) / ((np.exp((h * c)/(k_B * T * lambda_i * 1e-6)) - 1) * lambda_i**5) * 1e14

def nb_B(lambda_i, T):
    return (2 * c) / ((np.exp((h * c)/(k_B * T * lambda_i * 1e-6)) - 1) * lambda_i**4) * 1e8

wavelengths = torch.linspace(0.350, 3, 2651)
# Restrict wavelengths to avoid issues and sub-sample for speed.
wavelengths = wavelengths[(wavelengths != 0.5) & (wavelengths != 1.0)]
T_e = 2073.15  # Emitter temperature [K]
T_PV = 300     # PV temperature [K]
nb_B_e = nb_B(wavelengths, T_e)
nb_B_PV = nb_B(wavelengths, T_PV)

def IQE(wavelength, e_g):
    lambda_g = np.ceil(1240 / e_g) / 1000.0
    if (lambda_g > wavelengths[-1]):
        l_index = wavelengths[-1]
    else:
        l_index = torch.where(wavelength >= lambda_g)[0][0]
    IQE = torch.ones(len(wavelength))
    for i in range(l_index, len(wavelength)):
        IQE[i] = 0
    return IQE

def JV(em, IQE, lambda_i):
    em = em.squeeze()
    J_L = q * torch.sum(em * nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = q * torch.sum(nb_B_PV * IQE) * (lambda_i[1] - lambda_i[0])
    V_oc = (k_B * T_PV / q) * torch.log(J_L / J_0 + 1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc
    J = J_L - J_0 * (torch.exp(q * V / (k_B * T_PV)) - 1)
    P = V * J
    return torch.max(P)

def power_ratio(lambda_i, transmitted_power, T_emitter, E_g_PV):
    # transmitted_power acts as the emissivity dataset here
    P_emit = torch.sum(transmitted_power * Blackbody(lambda_i, T_emitter)) * (lambda_i[1] - lambda_i[0])
    IQE_PV = IQE(lambda_i, E_g_PV)
    JV_PV = JV(transmitted_power, IQE_PV, lambda_i)
    FOM = JV_PV / P_emit
    return FOM

# -----------------------
# Load material data (for dispersion)
# -----------------------
n_all = np.load('/home/rliacobacci/Downloads/n_allHTMats.npz')
k_all = np.load('/home/rliacobacci/Downloads/k_allHTMats.npz')
w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j

# Depth for the homogeneous layer (same as previous grating depth)
# aln_depth = 0.473

# Number of harmonics for S4 simulation
harmonics = 20

# -----------------------
# Simulation function
# -----------------------
def simulate_homogeneous_fom(random_factor, homogeneous = False):
    """
    Runs a homogeneous simulation over all wavelengths.
    The effective permittivity in the homogeneous layer is set via:
        effective_eps = random_factor * ((AlN_eps - Vacuum_eps)) + Vacuum_eps
    where for a baseline pure AlN one would have random_factor=1,
    and for pure vacuum random_factor=0.
    """
    aln_depth = random_factor[-1]
    random_factor = random_factor[:random_factor.shape[0]-1]
    transmitted_power_per_wavelength = np.zeros(len(wavelengths))
    for i, wavelength in enumerate(wavelengths):
        S = S4.New(Lattice=1, NumBasis=harmonics)
        # Define materials
        # Note: The material data (aln_n, w_n) is wavelength-dependent.
        S.SetMaterial(Name='AluminumNitride', Epsilon=(aln_n[i])**2)
        S.SetMaterial(Name='Tungsten', Epsilon=(w_n[i])**2)
        S.SetMaterial(Name='Vacuum', Epsilon=(1+0j)**2)
        
        # Top layer: AirAbove
        S.AddLayer(Name='AirAbove', Thickness=1, Material='Vacuum')
        
        # Homogeneous layer: set custom effective permittivity based on random_factor.
        # For each wavelength, a pure AlN layer would have permittivity (aln_n[i])**2,
        # and vacuum would be 1. Thus:
        # effective_eps = random_factor * ((aln_n[i])**2 - 1) + 1
        # S.SetMaterial(Name='Custom', Epsilon=effective_eps)
        if not homogeneous:
            S.AddLayer(Name = 'Grid', Thickness=aln_depth, Material = 'Vacuum')
            num_image_squares = random_factor.shape[0]
            for q in range(num_image_squares):
                S.SetMaterial(Name = f'Grid material {q}', Epsilon = random_factor[q] * (aln_n[i]**2 - 1) + 1)
                S.SetRegionRectangle(Layer = 'Grid', Material = f'Grid material {q}', Center = ((q+1)/num_image_squares - 1/(2*num_image_squares), .5), Halfwidths = (1/(2*num_image_squares), 1/2), Angle = 0)
        elif homogeneous == 1:
            S.AddLayer(Name = 'Grid', Thickness = .473, Material = 'AluminumNitride')
        elif homogeneous == 2:
            S.AddLayer(Name = 'Grid', Thickness = .473, Material = 'Vacuum')
        
        # Bottom layers
        S.AddLayer(Name='TungstenBelow', Thickness=1, Material='Tungsten')
        S.AddLayer(Name='AirBelow', Thickness=1, Material='Vacuum')
        
        # Excitation
        S.SetExcitationPlanewave(
            IncidenceAngles=(config['off_angle'], 0),
            sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2), Order=0)
        S.SetOptions(PolarizationDecomposition=True)
        S.SetFrequency(1 / float(wavelength))
        
        # Obtain transmitted power flux (from the bottom metal layer)
        (norm_forw, norm_back) = S.GetPowerFlux(Layer='AirAbove', zOffset=0)
        transmitted_power_per_wavelength[i] = 1 - np.abs(norm_back)
        
        del S
    transmitted_power = torch.tensor(transmitted_power_per_wavelength, dtype=torch.float32)
    fom = power_ratio(wavelengths, transmitted_power, T_e, 0.726)
    return fom.item()

# -----------------------
# Compute baseline FOM values
# -----------------------
print("Computing baseline FOM values ...")
num_image_squares = 100
fom_baseline_aln = simulate_homogeneous_fom(random_factor=np.ones(shape=(num_image_squares+1,)), homogeneous=1)    # Homogeneous AlN layer
fom_baseline_vac = simulate_homogeneous_fom(random_factor=np.zeros(shape=(num_image_squares+1,)), homogeneous=2)      # Homogeneous vacuum layer

with open(fom_file, 'w') as f:
    f.write("Baseline FOM values:\n")
    f.write(f"  Homogeneous AlN: {fom_baseline_aln:.6f}\n")
    f.write(f"  Homogeneous Vacuum: {fom_baseline_vac:.6f}\n")
    f.write("-" * 30 + "\n")

print(f"Baseline FOM (AlN): {fom_baseline_aln:.6f}")
print(f"Baseline FOM (Vacuum): {fom_baseline_vac:.6f}")

# -----------------------
# Main loop: generate random images until condition is met
# -----------------------
iteration = start_iteration
while True:
    # Generate a new random factor (a scalar between 0 and 1)
    random_factor = np.random.rand(num_image_squares+1,)
    
    # Calculate FOM for the random homogeneous structure
    fom_random = simulate_homogeneous_fom(random_factor)
    
    # log_line = f"Iteration {iteration}: random_factor = {random_factor:.4f}, FOM = {fom_random:.6f}"
    # print(log_line)
    # with open(fom_file, 'a+') as f:
    #     f.write(log_line + "\n")
    print(fom_random, random_factor[-1])
    # Optionally: Save the random factor as the "image" for this iteration
    image_filename = os.path.join(log_dir, f"iteration_{iteration}.npy")
    # np.save(image_filename, np.array([random_factor]))
    # print(f"Saved random image (factor) to {image_filename}")
    
    # Stop if the FOM exceeds both baseline values
    if (fom_random > fom_baseline_aln) and (fom_random > fom_baseline_vac):
        print("FOM exceeded both baseline values. Stopping random image generation.")
        with open(fom_file, 'a+') as f:
            f.write("Stopping condition met.\n")
            np.save(image_filename, np.array([random_factor]))
            print(f"Saved random image (factor) to {image_filename}")
        break
    
    iteration += 1
    # Optionally, add a delay or maximum iteration count to avoid infinite loops.
