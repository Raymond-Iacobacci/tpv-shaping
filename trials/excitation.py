#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import S4
import ff

def main():
    # ----------------------------
    # simulation parameters
    # ----------------------------
    L = 1.0               # lattice period (µm)
    N = 3                 # number of Fourier basis functions
    wavelength = 1.1      # wavelength (µm)
    vac_depth = 1.0       # thickness of vacuum layer (µm)
    h_index = -1          # diffraction order index
    amplitude = 1.0       # amplitude for the exterior excitation

    # sampling resolution
    n_x_pts = 300
    n_z_pts = 300

    # ----------------------------
    # build the S4 structure
    # ----------------------------
    S = S4.New(Lattice=L, NumBasis=N)
    S.SetMaterial(Name='Vac', Epsilon=1.0)
    S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[130]**2))
    S.SetMaterial(Name='W',   Epsilon=ff.w_n[130]**2)

    # layers: vacuum then dummy substrate
    S.AddLayer(Name='Vacuum', Thickness=vac_depth, Material='Vac')
    S.AddLayer(Name='Substrate', Thickness=1.0, Material='Vac')

    S.SetFrequency(1.0 / wavelength)

    # single y-polarized exterior harmonic
    # excitations = ((1, b'y', amplitude),(2, b'y', amplitude))
    excitations = ((2, b'y', amplitude),)

    S.SetExcitationExterior(Excitations=excitations)
    (forw_amp, back_amp) = S.GetAmplitudes('Vacuum', zOffset=1)
    print(forw_amp)
    print(back_amp)
    # S.SetExcitationPlanewave((0,0), sAmplitude=amplitude, pAmplitude=0, Order=0)

    # ----------------------------
    # sample the field over x and z
    # ----------------------------
    x_space = np.linspace(0, L, n_x_pts)
    z_space = np.linspace(0, vac_depth+1, n_z_pts)

    E_y = np.zeros((n_z_pts, n_x_pts), dtype=complex)
    for iz, z in enumerate(z_space):
        for ix, x in enumerate(x_space):
            # GetFields returns a tuple; [0][1] is the E_y component
            E_y[iz, ix] = S.GetFields(x, 0, z)[0][1]
    print(np.max(np.abs(E_y)))
    # ----------------------------
    # plot real & imaginary as 2D maps
    # ----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axs[0].imshow(
        np.real(E_y),
        extent=[0, L, 0, vac_depth],
        aspect='auto',
        origin='lower'
    )
    axs[0].set_title('Re [Eᵧ(x, z)]')
    axs[0].set_xlabel('x (µm)')
    axs[0].set_ylabel('z (µm)')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(
        np.imag(E_y),
        extent=[0, L, 0, vac_depth],
        aspect='auto',
        origin='lower'
    )
    axs[1].set_title('Im [Eᵧ(x, z)]')
    axs[1].set_xlabel('x (µm)')
    axs[1].set_ylabel('z (µm)')
    fig.colorbar(im1, ax=axs[1])

    plt.suptitle(f'Y-polarized order {h_index} in vacuum layer')
    plt.show()

if __name__ == '__main__':
    main()
