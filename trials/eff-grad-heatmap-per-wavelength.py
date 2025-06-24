# gradient_descent_on_grating.py

from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def assign_variables(**kwargs):
    """
    Yield the values of kwargs as a tuple, in the same order
    they were passed in (Python 3.6+ preserves kwargs order).

    Usage:
        with assign_variables(x=10, y=20) as (x, y):
            print(x, y)  # 10 20
    """
    # Grab the values in insertion order (guaranteed from Python 3.6+).
    values = tuple(kwargs.values())
    try:
        yield values
    finally:
        # No cleanup needed—locals created by the `as` clause
        # will simply go out of scope when the block ends.
        pass


def interpolate_dataset(data, extend=0, poly_order=1):
    """
    Interpolate a dataset of (x,y) pairs, where x are integers, using polynomial interpolation.

    Args:
        data: List of (x,y) tuples where x are integers
        extend: Number of points to extend beyond max(x)
        poly_order: Order of the polynomial to use for interpolation (1=linear, 2=quadratic, etc.)
                    Will automatically reduce order if not enough points are available

    Returns:
        List of (x,y) tuples with interpolated and extended values
    """
    # Sort data by x values
    sorted_data = sorted(data, key=lambda point: point[0])

    # Extract x and y values
    x_values = [point[0] for point in sorted_data]
    y_values = [point[1] for point in sorted_data]

    # Find min and max x values
    min_x = min(x_values)
    max_x = max(x_values)

    # Create a dictionary from original data for easy lookup
    data_dict = {x: y for x, y in sorted_data}

    # Ensure we have enough points for the polynomial order
    actual_poly_order = min(poly_order, len(sorted_data) - 1)
    if actual_poly_order != poly_order:
        print(f"Warning: Reduced polynomial order from {poly_order} to {actual_poly_order} due to insufficient data points")

    result = []

    # If we have enough points for polynomial interpolation
    if len(sorted_data) > 1:
        if actual_poly_order == 1:
            # Linear interpolation
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    result.append((x, data_dict[x]))
                else:
                    # Find closest points before and after x
                    left_idx = 0
                    while left_idx < len(x_values) - 1 and x_values[left_idx + 1] <= x:
                        left_idx += 1

                    right_idx = len(x_values) - 1
                    while right_idx > 0 and x_values[right_idx - 1] >= x:
                        right_idx -= 1

                    x1, y1 = x_values[left_idx], y_values[left_idx]
                    x2, y2 = x_values[right_idx], y_values[right_idx]
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    result.append((x, y))
        else:
            # Polynomial interpolation
            coeffs = np.polyfit(x_values, y_values, actual_poly_order)
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    result.append((x, data_dict[x]))
                else:
                    y = np.polyval(coeffs, x)
                    result.append((x, y))
    else:
        # Just one point, can't interpolate
        result = sorted_data.copy()

    # Extend beyond max_x if requested (always using linear extension)
    if extend > 0 and len(x_values) >= 2:
        second_to_last_x = x_values[-2]
        second_to_last_y = y_values[-2]
        last_x = x_values[-1]
        last_y = y_values[-1]
        slope = (last_y - second_to_last_y) / (last_x - second_to_last_x)
        for i in range(1, extend + 1):
            x = max_x + i
            y = last_y + slope * i
            result.append((x, y))

    return result


def save_checkpoint(path, epoch, gratings, optimizer, grating_epochs):
    checkpoint = {
        'epoch': epoch,
        'gratings': [g.detach().cpu() for g in gratings],
        'optimizer': optimizer.state_dict(),
        'grating_epochs': grating_epochs,
        # NEW ----------------------------
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
    }
    torch.save(checkpoint, path)


'''
This file originally did direct gradient descent on a bilayer system.
We now only compute, for each (a,b) in [0,1]×[0,1], the efficiency and its gradient with respect to a and b,
and log them into numpy arrays. No optimization loop is performed.
'''

import math

class SwingLR:
    """
    Adaptive LR scheduler that *boosts* the LR after a run of improvements
    and *cuts* it sharply after a run of regressions.
    (Not used in the sweeping version, but retained for completeness.)
    """
    def __init__(self,
                 optimizer,
                 mode: str = 'max',
                 factor_up: float = 1.25,
                 factor_down: float = 0.2,
                 patience_up: int = 3,
                 patience_down: int = 2,
                 threshold: float = 1e-5,
                 min_lr: float = 1e-6,
                 max_lr: float = 1.0,
                 cooldown: int = 0):
        if factor_up <= 1.0: raise ValueError("factor_up must be > 1.0")
        if factor_down >= 1.0: raise ValueError("factor_down must be < 1.0")

        self.opt = optimizer
        self.mode = mode
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.patience_up = patience_up
        self.patience_down = patience_down
        self.threshold = threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cooldown = cooldown

        self._best = None
        self._num_up = 0
        self._num_down = 0
        self._wait = 0  # cooldown counter

    def _is_better(self, metric):
        if self._best is None:  # first call
            return True
        delta = metric - self._best
        if self.mode == 'max':
            return delta > self.threshold
        else:  # 'min'
            return delta < -self.threshold

    def _apply_change(self, factor):
        for pg in self.opt.param_groups:
            new_lr = pg['lr'] * factor
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))
            pg['lr'] = new_lr

    def step(self, metric):
        """
        Call *once per epoch* with the latest FOM (higher is better if mode='max').
        (Not used in sweep mode.)
        """
        if self._wait > 0:
            self._wait -= 1
            return

        better = self._is_better(metric)
        if better:
            self._best = metric
            self._num_up += 1
            self._num_down = 0
            if self._num_up >= self.patience_up:
                self._apply_change(self.factor_up)
                self._num_up = 0
                self._wait = self.cooldown
        else:
            self._num_down += 1
            self._num_up = 0
            if self._num_down >= self.patience_down:
                self._apply_change(self.factor_down)
                self._num_down = 0
                self._wait = self.cooldown


class ManualOptimizer:
    def __init__(self, lr):
        # SwingLR just looks for opt.param_groups[…]['lr']
        self.param_groups = [{'lr': lr}]


class ManualSwingLR(SwingLR):
    def __init__(self, initial_lr, **kw):
        super().__init__(optimizer=ManualOptimizer(initial_lr), **kw)

    @property
    def lr(self):
        return self.opt.param_groups[0]['lr']


sched = ManualSwingLR(1,
                      mode='max',
                      factor_up=2**(1/3),
                      factor_down=0.2**0.5,
                      patience_up=1,
                      patience_down=1,
                      cooldown=1,
                      min_lr=1,
                      max_lr=100.0)

import json
import os
import random
import secrets
import string
import sys
from pathlib import Path

import ff
import matplotlib.pyplot as plt
import S4
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm

config = {
    "seeds": {
        "torch": int(92),  # 52
        "numpy": int(81),  # 41
        "random": int(70)  # 30
    },
    "n_incidence_harmonics": int(30),
    "n_grating_harmonics": int(30),
    "n_grating_elements": int(20),
    "ang_polarization": float(0),
    "learning_rate": float(100),
    "n_gratings": int(1),
    "bias": 1e-3,
    "logging": False,
    "optimizer": False
}

checkpoint = False
if config["logging"]:
    root_log_dir = os.path.join(ff.home_directory(), 'logs', 'gradient-descent-on-grating-t')
    os.makedirs(root_log_dir, exist_ok=True)
    if not checkpoint:
        log_dir = ff.get_unique_log_dir(root_log_dir, config)
        checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
        start_epoch = 0
        grating_epochs = [0] * config['n_gratings']
    else:
        log_dir = "/home/rliacobacci/tpv-shaping/logs/gradient-descent-on-grating-t/def-97060e69_5"
        checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
        chk = torch.load(checkpoint_path)

        torch.set_rng_state(chk['torch_rng'])
        torch.cuda.set_rng_state_all(chk['cuda_rng'])
        np.random.set_state(chk['numpy_rng'])
        random.setstate(chk['python_rng'])

        start_epoch = chk['epoch'] + 1
        gratings = [torch.nn.Parameter(torch.empty_like(t)) for t in chk['gratings']]
        for p, saved in zip(gratings, chk['gratings']):
            p.data.copy_(saved)
        grating_epochs = chk['grating_epochs']

    print(f'Log directory: {log_dir}')
    config_file = os.path.join(log_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    fom_file = os.path.join(log_dir, 'fom_values.txt')
    image_file = os.path.join(log_dir, 'image_values.txt')
else:
    start_epoch = 0

if not checkpoint:
    torch.manual_seed(config['seeds']['torch'])
    np.random.seed(config['seeds']['numpy'])
    random.seed(config['seeds']['random'])

wavelengths = torch.linspace(.35, 3, 2651)
exclude_wavelengths = torch.tensor([.5, 1])

y_pt = 0
x_density = 5
n_x_measurement_pts = x_density * config['n_grating_elements']
L = 1.1
z_buff = .15


def make_measurement_grid(L, n_cells, k):
    """
    Return the x–coordinates of k measurements per cell.

    Parameters
    ----------
    L        : total length
    n_cells  : number of unit cells (your n_x_measurement_pts)
    k        : points per cell (k=1 → 0.5; k=2 → 0.25,0.75; k=3 → 1/6,1/2,5/6 …)
    """
    dx = L / n_cells                              # cell width
    fractions = np.arange(1, 2 * k, 2) / (2 * k)  # shape (k,)
    starts = np.arange(n_cells)[:, None]          # shape (n_cells,1)
    return ((starts + fractions) * dx).ravel()    # flatten to 1-D


x_measurement_space = make_measurement_grid(L=L, n_cells=x_density, k=config['n_grating_elements'])
depth = .473
z_space = np.linspace(0, .5 + depth + 1, 40)
z_measurement_space = z_space[(z_space >= .5) & (z_space <= .5 + depth)]
jump = 10


def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float, index: int):
    """
    Compute the gradient of the FOM wrt each grating element, and return (dfom_deps, fom).
    - grating: torch.Tensor of length 20, each in [0,1].
    - L: total grating length.
    - ang_pol: polarization angle for excitation.
    - index: just for printing/logging.

    Returns:
        dfom_deps: 1-D torch.Tensor of length 20: ∂FOM/∂(each grating element)
        fom:      scalar torch.Tensor = efficiency
    """
    transmitted_power = []
    indices_used = []
    sample_mask = torch.zeros_like(wavelengths, dtype=torch.bool)
    dfom_deps = torch.zeros((config['n_grating_elements']))
    dflux_deps = torch.zeros((wavelengths.shape[0], n_x_measurement_pts))

    for i_wl, wl in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave=False)):
        if wl.item() in exclude_wavelengths or i_wl % jump != 0:
            continue
        indices_used.append(i_wl)
        sample_mask[i_wl] = True
        S = S4.New(Lattice=L, NumBasis=config['n_grating_harmonics'])
        S.SetMaterial(Name='W', Epsilon=ff.w_n[i_wl + 130]**2)
        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=ff.aln_n[i_wl + 130]**2)

        S.AddLayer(Name='VacuumAbove', Thickness=.5, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='Vac')

        for ns in range(grating.shape[-1]):
            S.SetMaterial(
                Name=f'sq{ns+1}',
                Epsilon=grating[ns].item() * (ff.aln_n[i_wl + 130]**2 - 1) + 1
            )
            S.SetRegionRectangle(
                Layer='Grating',
                Material=f'sq{ns+1}',
                Center=(((ns + 1) / grating.shape[-1] - 1 / (2 * grating.shape[-1])) * L, .5),
                Halfwidths=((1 / (2 * grating.shape[-1]) * L), .5),
                Angle=0
            )

        S.AddLayer(Name='Ab', Thickness=1, Material='W')
        S.SetFrequency(1 / wl)

        S_adj = S.Clone()

        S.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),
            sAmplitude=np.cos(ang_pol * np.pi / 180),
            pAmplitude=np.sin(ang_pol * np.pi / 180),
            Order=0
        )
        (_, back) = S.GetPowerFlux(Layer='VacuumAbove', zOffset=0)
        back_t = torch.as_tensor(back).abs()
        transmitted_power.append(1 - back_t)

        adj_fields = np.zeros((1, 1, n_x_measurement_pts), dtype=complex)
        for ix, x in enumerate(x_measurement_space):
            adj_fields[0, 0, ix] = S.GetFields(x, y_pt, z_buff)[0][1]
        grating_fields = np.zeros((z_measurement_space.shape[0], 1, x_measurement_space.shape[0], 3), dtype=complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                grating_fields[iz, 0, ix] = S.GetFields(x, y_pt, z)[0]
        del S

        adj_excitation_mag = np.abs(adj_fields[0, 0, :]**2)
        adj_excitation_phase = np.angle(adj_fields[0, 0, :])
        adj_excitation = adj_excitation_mag * np.exp(1j * adj_excitation_phase)

        fourier_adj_excitation = ff.create_step_excitation(
            basis=S_adj.GetBasisSet(),
            step_values=adj_excitation,
            num_harmonics=config['n_incidence_harmonics'],
            x_shift=0,
            initial_phase=0,
            amplitude=1.0,
            period=L,
            plot_fourier=False
        )

        S_adj.SetExcitationExterior(Excitations=tuple(fourier_adj_excitation))

        adj_grating_fields = np.zeros((z_measurement_space.shape[0], 1, x_measurement_space.shape[0], 3), dtype=complex)
        for iz, z in enumerate(z_measurement_space):
            for ix, x in enumerate(x_measurement_space):
                adj_grating_fields[iz, y_pt, ix] = S_adj.GetFields(x, y_pt, z)[0]
        dflux_deps_wl = 2 * (wl / 1e6) ** -2 * ff.e_0 * torch.real(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(grating_fields),
                         torch.as_tensor(adj_grating_fields).conj())
        )
        dflux_deps[i_wl] = torch.mean(dflux_deps_wl, dim=0).squeeze() * (ff.aln_n[i_wl + 130]**2 - 1)
        del S_adj

    dflux_deps = torch.tensor(dflux_deps, requires_grad=True)
    data = [(indices_used[i], transmitted_power[i]) for i in range(len(indices_used))]
    interpolated_data = interpolate_dataset(data, extend=0, poly_order=1)
    interpolated_ppw = torch.tensor([i[1] for i in interpolated_data], requires_grad=True)
    interpolated_ppw.retain_grad()
    transmitted_power = torch.tensor(transmitted_power, requires_grad=True)

    idx = sample_mask.nonzero(as_tuple=True)[0]
    fom = ff.power_ratio(wavelengths[idx], interpolated_ppw[idx], ff.T_e, .726)
    fom.backward()

    dfom_dflux = interpolated_ppw.grad

    G = dfom_dflux[idx].unsqueeze(1) * dflux_deps[idx]
    dwl = jump * (wavelengths[1] - wavelengths[0])
    dfom_deps = torch.mean(G, dim=0) * dwl
    dfom_deps = dfom_deps.reshape(-1, x_density).mean(axis=1).detach()

    # ────────────────────────────────────────────────────────
    # now compute ∂FOM/∂a and ∂FOM/∂b *for each sampled wavelength*
    n_elem   = config['n_grating_elements']
    # G has shape (n_wl_sampled, n_x_pts) → group into elements:
    G_elem   = G.reshape(-1, n_elem, x_density).mean(dim=2)
    # average first 18 for “a”, last 2 for “b”
    grad_a_wl = G_elem[:, :18].mean(dim=1).detach().cpu().numpy()
    grad_b_wl = G_elem[:, 18:20].mean(dim=1).detach().cpu().numpy()
    # ────────────────────────────────────────────────────────

    print(f'Index: {index}, efficiency: {fom}')
    print(f'Grating used: {[round(i.item() * 1000) / 1000 for i in grating]}')

    if config['logging']:
        with open(fom_file, 'a+') as f:
            f.write(str(f'{index}:{fom.item()}\n'))
        with open(image_file, 'a+') as f:
            for i in grating:
                f.write(str(f'{i.item()} '))
                f.write("\n")

    # return dfom_deps, fom.detach()
    return (
        dfom_deps,
        fom.detach(),
        dfom_dflux.detach().cpu().numpy(),
        dflux_deps.detach().cpu().numpy(),
        transmitted_power.detach().cpu().numpy(),
        np.array(indices_used, dtype=np.int64),
        grad_a_wl,
        grad_b_wl,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Replace the original gradient‐descent loop with a resumable sweep over (a,b).
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as _np
    import torch as _torch
    import os

    # where to dump each (a,b) folder
    RESULTS_ROOT = "results"
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # 0a) our “heatmap” file
    HEATMAP_FILE = "heatmaps.npz"

    # 1) grid of values from 0 to 1 in increments of 0.05


    a_vals = _np.arange(0.0, 1.0001, 0.05)
    b_vals = _np.arange(0.0, 1.0001, 0.05)
    Na, Nb = len(a_vals), len(b_vals)

    # 2) Check if a previous .npz exists. If so, load it. Otherwise, initialize as NaN.
    # if os.path.exists("sweep_results.npz"):
    #     data = _np.load("sweep_results.npz")
    if os.path.exists(HEATMAP_FILE):
        data = _np.load(HEATMAP_FILE)
        effs   = data["effs"]    # shape (Na, Nb)
        grad_a = data["grad_a"]  # shape (Na, Nb)
        grad_b = data["grad_b"]  # shape (Na, Nb)
        # a_vals and b_vals are the same arrays; re‐load to ensure consistency
        a_vals = data["a_vals"]
        b_vals = data["b_vals"]
    else:
        # Create arrays filled with NaN so we can detect "missing" entries
        effs   = _np.full((Na, Nb), _np.nan, dtype=_np.float64)
        grad_a = _np.full((Na, Nb), _np.nan, dtype=_np.float64)
        grad_b = _np.full((Na, Nb), _np.nan, dtype=_np.float64)

    # 3) Fixed parameters for gradient_per_image (same as before)
    L       = L  # total grating length (already defined above)
    ang_pol = config["ang_polarization"]

    # 4) Loop over all (i,j) pairs. Skip any (i,j) that is not NaN (i.e., already computed).
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            # If this point was already computed, skip it
            if not _np.isnan(effs[i, j]):
                continue

            # 4a) build a 20‐element torch.Tensor: indices 0–17 all = a; indices 18–19 all = b
            grating_vals = _torch.tensor([a] * 18 + [b] * 2, dtype=_torch.float32)

            # 4b) call gradient_per_image → (dfom_deps, fom)
            dfom_deps, fom, dfom_dflux, dflux_deps, p_trans, idx_used, grad_a_wl, grad_b_wl = gradient_per_image(                grating_vals,
                L,
                ang_pol,
                index=0
            )

            # 4c) extract gradient wrt 'a' and 'b'
            effs[i, j]   = fom.cpu().item()
            grad_vals_np = dfom_deps.cpu().numpy()
            grad_a[i, j] = grad_vals_np[0:18].mean()
            grad_b[i, j] = grad_vals_np[18:20].mean()

            # 4d) overwrite the .npz after each new point, preserving prior data
            _np.savez(
                HEATMAP_FILE,
                effs=effs,
                grad_a=grad_a,
                grad_b=grad_b,
                a_vals=a_vals,
                b_vals=b_vals
            )

            # 4e) dump per-(a,b) raw gradients and power into its own folder
            subdir = os.path.join(RESULTS_ROOT, f"a_{a:.2f}_b_{b:.2f}")
            os.makedirs(subdir, exist_ok=True)
            dflux_deps_sampled = dflux_deps[idx_used]
            _np.savez(
                os.path.join(subdir, "per_wavelength_data.npz"),
                dfom_dflux = dfom_dflux,            # shape (n_sampled_wls,)
                dflux_deps = dflux_deps_sampled,    # shape (n_sampled_wls, n_x_pts)
                P_trans    = p_trans,          # P(λ)
                indices    = idx_used,         # which wavelength‐indices
                wavelengths= wavelengths.numpy()[idx_used],
                x_space    = x_measurement_space,
                z_space    = z_measurement_space,
                grad_a_wl  = grad_a_wl,
                grad_b_wl  = grad_b_wl,
            )

            print(
                f"Swept a={a:.2f}, b={b:.2f} → "
                f"eff={effs[i,j]:.6f}, ∂eff/∂a={grad_a[i,j]:.6e}, ∂eff/∂b={grad_b[i,j]:.6e}"
            )

    print("Sweep complete (or resumed). All data saved to 'sweep_results.npz'.")
