"""
This file simulates the optimization of images without an explicit generator, i.e. the images themselves are optimized.
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

timing_field_queries = False

config = {
    "num_images": int(1),
    "seeds": {
        "torch": int(52),
        "numpy": int(41)
    },
    "learning_rate": float(1e12),
    "incidence_angle": float(0),
    "excite_harmonics": int(100),
    "image_harmonics": int(14),
    "num_image_squares": int(10),
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

num_cycles = 400

vacuum_thickness = .5 # microns
grid_thickness = .473
absorb_thickness = 1

y_pt = 0
z_dots, y_dots, x_dots, x_msrmt_dots, num_image_squares = 100, 1, 100, 50, int(config['num_image_squares'])
z_space = np.linspace(0, vacuum_thickness+grid_thickness+absorb_thickness, z_dots)
z_grtg_space = z_space[(z_space >= vacuum_thickness) & (z_space <= vacuum_thickness + grid_thickness)]
z_buffer = .15 # This is for when we offset the measurement later on
y_space = np.linspace(y_pt, y_pt, y_dots)
x_space = np.linspace(0, 1, x_dots) # Used for plotting
x_msrmt_space = np.array([(q+1)/x_msrmt_dots - 1/(2*x_msrmt_dots) for q in range(x_msrmt_dots)])
x_grtg_space = np.array([(q+1) / num_image_squares - 1 / (2 * num_image_squares) for q in range(num_image_squares)])

generated_images = (torch.rand((config['num_images'], num_image_squares), requires_grad = False))
homogeneous = False

for it in range(num_cycles):

    with open(fom_file, 'a+') as f:
        f.write(f'\nIteration {it}\n')
        f.write('-' * 30 + '\n')

    dfom_deps = torch.zeros((num_image_squares,))
    
    for image_index, image in enumerate(generated_images):
        image.requires_grad_(True)
        
        # Initialize wavelength-specific variables:
        transmitted_power_per_wavelength = torch.zeros((len(wavelengths),))
        dflux_deps_all_wavelength = np.zeros((wavelengths.shape[0], num_image_squares))
        
        for i_wavelength, wavelength in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
            if timing_field_queries: wavelength_process_time_start = time.time()
            S = S4.New(Lattice = 1, NumBasis=config['image_harmonics']) # This sets nonzero harmonics only in a single (x) direction, so we actually have 14 harmonics in that direction
            S.SetOptions(LanczosSmoothing=True)

            S.SetMaterial(Name='W', Epsilon=(ff.w_n[i_wavelength])**2)
            S.SetMaterial(Name='Vacuum', Epsilon=(1 + 0j)**2)
            S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wavelength])**2)

            S.AddLayer(Name = 'VacuumAbove', Thickness = vacuum_thickness, Material = 'Vacuum')
            if homogeneous:
                S.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'AlN')
            else:
                S.AddLayer(Name = 'Grid', Thickness = grid_thickness, Material = 'Vacuum')

                for q in range(num_image_squares):
                    S.SetMaterial(Name = f'Grid material {q}', Epsilon = image[q].item() * (ff.aln_n[i_wavelength]**2 - 1) + 1)
                    S.SetRegionRectangle(Layer = 'Grid', Material = f'Grid material {q}', Center = ((q+1)/num_image_squares - 1/(2*num_image_squares), .5), Halfwidths = (1/(2*num_image_squares), 1/2), Angle = 0)

            S.AddLayer(Name = 'Absorber', Thickness = absorb_thickness, Material = 'W')

            S.SetFrequency(1 / wavelength)

            S2 = S.Clone() # Test this by removing the excitation from the cloned simulation and seeing if there's fields anywhere

            S.SetExcitationPlanewave(IncidenceAngles=(config['incidence_angle'], 0), sAmplitude=1/np.sqrt(2), pAmplitude=1/np.sqrt(2), Order=0) # In the real simulation this should be edited to have 1/sqrt(2) amplitude in both directions

            (forw, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0)
            
            transmitted_power_per_wavelength[i_wavelength] = 1 - np.abs(back)
            if homogeneous: continue
            if timing_field_queries: fields_time_start = time.time()
            adj_src_fields = np.zeros((1, y_dots, x_msrmt_dots), dtype = complex)
            if timing_field_queries: fields_time = 0
            for ix, x in enumerate(x_msrmt_space):
                if timing_field_queries: start = time.time()
                adj_src_fields[0, 0, ix] = S.GetFields(x, y_pt, z_buffer)[0][0] # This computes the electric field. The .25 buffer is to handle edge effects. To obtain the z-flux of the electric field, we need to store the first element of the last dimension via the RHR.
                if timing_field_queries: fields_time += time.time() - start
            if timing_field_queries: print(f"First GetFields loop took: {fields_time:.2f} seconds")
            grtg_fields = np.zeros((z_grtg_space.shape[0], y_dots, x_grtg_space.shape[0], 3), dtype = complex)
            if timing_field_queries: fields_time = 0
            for iz, z in enumerate(z_grtg_space):
                for ix, x in enumerate(x_grtg_space):
                    if timing_field_queries: start = time.time()
                    grtg_fields[iz, y_pt, ix] = S.GetFields(x, y_pt, z)[0]
                    if timing_field_queries: fields_time += time.time() - start
            if timing_field_queries: print(f"Second GetFields loop took: {fields_time:.2f} seconds")
            cartesian_excitation_magnitude = np.mean(np.abs(adj_src_fields[0, 0, :]))
            cartesian_excitation_phase = np.angle(adj_src_fields[0, 0, :])
            cartesian_excitations = cartesian_excitation_magnitude * np.exp(1j * cartesian_excitation_phase)
            
            excitations = ff.create_step_excitation(basis = S2.GetBasisSet(), step_values = cartesian_excitations, num_harmonics = config['excite_harmonics'], x_shift = 0, initial_phase = 0, amplitude = 1.0, plot_fourier=False)
            
            S2.SetExcitationExterior(Excitations = tuple(excitations))

            adj_grtg_fields = np.zeros((z_grtg_space.shape[0], y_dots, x_grtg_space.shape[0], 3), dtype = complex)
            if timing_field_queries: fields_time = 0
            for iz, z in enumerate(z_grtg_space):
                z -= z_buffer
                for ix, x in enumerate(x_grtg_space):
                    if timing_field_queries: start = time.time()
                    adj_grtg_fields[iz, y_pt, ix] = S2.GetFields(x, y_pt, z)[0]
                    if timing_field_queries: fields_time += time.time() - start
            if timing_field_queries: print(f"Third GetFields loop took: {fields_time:.2f} seconds") 
            if timing_field_queries: total_fields_time = time.time() - fields_time_start
            if timing_field_queries: print(f"Total time for all GetFields operations: {total_fields_time:.2f} seconds")
            # Gradient calculations WRT FOM
            dflux_deps = 2 * wavelength ** 2 * ff.e_0 * np.real(np.einsum('ijkl,ijkl->ijk', grtg_fields, np.conjugate(adj_grtg_fields))) # The negative sign and the 2 in the adjoint source construction can factor out of the real function and the einsum function
            dflux_deps_all_wavelength[i_wavelength] = torch.mean(dflux_deps, dim = 0).squeeze() # This removes the z-dimension and the collapsed y-dimension from this simulation, giving us a gradient of dimension 1 / num_image_squares
            if timing_field_queries: wavelength_process_time = time.time() - wavelength_process_time_start
            if timing_field_queries: print(f"Full wavelength processing took: {wavelength_process_time:.2f} seconds")
            del S2
            del S
            gc.collect()
            
        dflux_deps_all_wavelength = torch.tensor(dflux_deps_all_wavelength, requires_grad = True)

        # Need to add scaling binary constraint here
        transmitted_power_per_wavelength = torch.tensor(transmitted_power_per_wavelength, requires_grad=True)
        fom = ff.power_ratio(wavelengths, transmitted_power_per_wavelength, ff.T_e, .726)
        with open(fom_file, 'a+') as f:
            f.write(f'Image {image_index} FOM: {fom.item()}\n')
        
        fom.backward()

        dfom_dflux = transmitted_power_per_wavelength.grad
        dfom_deps += torch.mean(dfom_dflux.unsqueeze(1).expand(wavelengths.shape[0], num_image_squares) * dflux_deps_all_wavelength, dim = 0)
        print(f'FOM: {fom.item()}, gradient avg: {torch.mean(torch.abs(dfom_deps)).item()}')
        print(f'Image: {generated_images}, gradient: {dfom_deps}')

    # Read the live gradient scale from file
    live_gradient_scale = 1.0  # Default value if file doesn't exist
    live_gradient_scale_file = Path(ff.home_directory()) / '.LIVE_GRADIENT_SCALE.txt'
    if live_gradient_scale_file.exists():
        try:
            with open(live_gradient_scale_file, 'r') as f:
                live_gradient_scale = float(f.read().strip())
        except (ValueError, FileNotFoundError):
            print("Error reading .LIVE_GRADIENT_SCALE.txt, using default value of 1.0")

    corrective_factor = config['learning_rate'] * (1+(it+1)/num_cycles) * live_gradient_scale
    print(f'Corrected gradient: {dfom_deps*corrective_factor}')
    generated_images -= dfom_deps.repeat(config['num_images'], 1)/config['num_images']*corrective_factor # Should we add a negative here, like in the original material?
    generated_images = torch.clamp(generated_images, 0, 1)

    image_file = os.path.join(log_dir, f'image_values_iteration_{it}.txt')
    images_numpy = generated_images.detach().numpy()
    # Save both the images and the gradient scale in the same file
    np.savez(
        image_file,
        images=images_numpy,
        gradient_scale=live_gradient_scale
    )


    # image_file = os.path.join(log_dir, f'image_values_iteration_{it}.txt')
    # with open(image_file, 'w') as f:
    #     for idx, image in enumerate(generated_images):
    #         f.write(f'{",".join(map(str, image.detach().numpy()))}\n')