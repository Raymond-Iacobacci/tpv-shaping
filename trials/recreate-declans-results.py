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
    # Grab the values in insertion order (guaranteed from Python 3.6+). :contentReference[oaicite:0]{index=0}
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
            # Linear interpolation (original method)
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
                    
                    # Interpolate y value
                    x1, y1 = x_values[left_idx], y_values[left_idx]
                    x2, y2 = x_values[right_idx], y_values[right_idx]
                    
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    result.append((x, y))
        else:
            # Polynomial interpolation
            coeffs = np.polyfit(x_values, y_values, actual_poly_order)
            
            # Interpolate for all integer points
            for x in range(min_x, max_x + 1):
                if x in data_dict:
                    # Use original data point if available
                    result.append((x, data_dict[x]))
                else:
                    # Use polynomial to interpolate
                    y = np.polyval(coeffs, x)
                    result.append((x, y))
    else:
        # Just one point, can't interpolate
        result = sorted_data.copy()
    
    # Extend beyond max_x if requested (always using linear extension)
    if extend > 0 and len(x_values) >= 2:
        # Get the last two points to determine the slope for extension
        second_to_last_x = x_values[-2]
        second_to_last_y = y_values[-2]
        last_x = x_values[-1]
        last_y = y_values[-1]
        
        # Calculate slope
        slope = (last_y - second_to_last_y) / (last_x - second_to_last_x)
        
        # Add extended points
        for i in range(1, extend + 1):
            x = max_x + i
            y = last_y + slope * i
            result.append((x, y))
    
    return result


def save_checkpoint(path, epoch, gratings, optimizer, grating_epochs):
    checkpoint = {
        'epoch'          : epoch,
        'gratings'       : [g.detach().cpu() for g in gratings],
        'optimizer'      : optimizer.state_dict(),
        'grating_epochs' : grating_epochs,
        # NEW ----------------------------
        'torch_rng'      : torch.get_rng_state(),
        'cuda_rng'       : torch.cuda.get_rng_state_all(),
        'numpy_rng'      : np.random.get_state(),
        'python_rng'     : random.getstate(),
    }
    torch.save(checkpoint, path)

'''
This file does direct gradient descent on a bilayer system.
The bilayer system has a base of homogeneous Tungsten and a top of a CaO grating, randomly generated at the start.
The bug came from the differing spatial period when used with the ff code. Fix that [ff code].
'''

import math

class SwingLR:
    """
    Adaptive LR scheduler that *boosts* the LR after a run of improvements
    and *cuts* it sharply after a run of regressions.

    Parameters
    ----------
    optimizer           : torch.optim.Optimizer
    mode                : 'max'  (improvement == larger metric) or 'min'
    factor_up           : multiplicative LR increase  (e.g. 1.25)
    factor_down         : multiplicative LR decrease  (e.g. 0.2)
    patience_up         : #consecutive better epochs required to raise LR
    patience_down       : #consecutive worse  epochs required to lower LR
    threshold           : tiny change treated as "no change"
    min_lr, max_lr      : hard bounds on LR
    cooldown            : #epochs to wait after a *change* before next change
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

        if factor_up   <= 1.0: raise ValueError("factor_up must be > 1.0")
        if factor_down >= 1.0: raise ValueError("factor_down must be < 1.0")

        self.opt              = optimizer
        self.mode             = mode
        self.factor_up        = factor_up
        self.factor_down      = factor_down
        self.patience_up      = patience_up
        self.patience_down    = patience_down
        self.threshold        = threshold
        self.min_lr           = min_lr
        self.max_lr           = max_lr
        self.cooldown         = cooldown

        self._best    = None
        self._num_up  = 0
        self._num_down= 0
        self._wait    = 0     # cooldown counter

    # ------------------------------------------------------------------ #
    def _is_better(self, metric):
        if self._best is None:            # first call
            return True
        delta = metric - self._best
        if self.mode == 'max':
            return delta >  self.threshold
        else:   # 'min'
            return delta < -self.threshold

    # ------------------------------------------------------------------ #
    def _apply_change(self, factor):
        for pg in self.opt.param_groups:
            new_lr = pg['lr'] * factor
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))
            pg['lr'] = new_lr

    # ------------------------------------------------------------------ #
    def step(self, metric):
        """
        Call *once per epoch* with the latest FOM (higher is better if mode='max').
        """
        if self._wait > 0:                # in cooldown, just count down
            self._wait -= 1
            return

        better = self._is_better(metric)

        if better:
            self._best = metric
            self._num_up  += 1
            self._num_down = 0
            if self._num_up >= self.patience_up:
                self._apply_change(self.factor_up)
                self._num_up = 0
                self._wait = self.cooldown
        else:
            self._num_down += 1
            self._num_up   = 0
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
plotting = True
if plotting:
    plt.ion()
import S4
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm

config = {
    "seeds": {
        "torch": int(92),#52
        "numpy": int(81),#41
        "random": int(70)#30
    },
    "n_grating_harmonics": int(1), # >= config['n_incidence_harmonics'], this tells you how many you need to set
    "ang_polarization": float(45), # Better angle for optimization (generally good at showing what can happen in two dimensions with unpolarized metamaterials)
    "logging": True,
}
if config["logging"]:
    root_log_dir = os.path.join(ff.home_directory(), 'logs', 'gradient-descent-on-grating-t')
    os.makedirs(root_log_dir, exist_ok = True)
    log_dir = ff.get_unique_log_dir(root_log_dir, config)

    print(f'Log directory: {log_dir}')
    config_file = os.path.join(log_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

wavelengths = torch.linspace(.22, 5, 4781)
exclude_wavelengths = torch.tensor([.5, 1])


L=1.10
jump = 1
depths1=[.473]
periods = [1]
def gradient_per_image(depth1, period):

    transmitted_power = []
    indices_used = []
    sample_mask = torch.zeros_like(wavelengths, dtype=torch.bool)

    if plotting:
        plt.close('all')
        fig, ax = plt.subplots()
        ax.set_xlabel("Wavelength index")
        ax.set_ylabel("Transmitted power")
        ax.set_title("Live emissivity (raw)")

    for i_wl, wl in enumerate(tqdm(wavelengths, desc="Processing wavelengths", leave = False)):
        if wl.item() in exclude_wavelengths or i_wl % jump != 0:
            continue
        indices_used.append(i_wl)
        sample_mask[i_wl] = True
        S = S4.New(Lattice = period, NumBasis = config['n_grating_harmonics'])
        S.SetMaterial(Name = 'W', Epsilon = ff.w_n[i_wl]**2) # This is extraordinarily necessary
        S.SetMaterial(Name = 'Vac', Epsilon = 1)
        S.SetMaterial(Name = 'AlN', Epsilon = ff.aln_n[i_wl]**2)
        S.AddLayer(Name = 'VacuumAbove', Thickness = .5, Material = 'Vac')
        S.AddLayer(Name = 'Grating', Thickness = depth1, Material = 'Vac')

        S.AddLayer(Name = 'Sst', Thickness = depth1, Material = 'AlN')
        S.AddLayer(Name = 'Ab2', Thickness = 1, Material = 'W')

        S.SetFrequency(1 / wl)

        S.SetExcitationPlanewave(IncidenceAngles = (0, 0), sAmplitude=np.cos(config['ang_polarization']*np.pi/180), pAmplitude=np.sin(config['ang_polarization']*np.pi/180), Order=0)
        (_, back) = S.GetPowerFlux(Layer = 'VacuumAbove', zOffset = 0) # We don't need to set the zOffset to the z_buff value because the power flux is the same through both points. The angle is different, yes, but that is not what is being measured in this line.
        if np.abs(back) > 1 or np.abs(1 + back) > 1:
            print(back, i_wl)
        back_t = torch.as_tensor(back).abs()
        transmitted_power.append(1 - back_t)

        del S

        if plotting:
            ax.clear()
            ax.plot(indices_used, transmitted_power)
            ax.set_xlabel("Wavelength index")
            ax.set_ylabel("Transmitted power")
            ax.set_title("Live emissivity (raw)")
            plt.pause(0.01)

    if plotting:
        plt.ioff()
        plt.close(fig)

    data = [(indices_used[i], transmitted_power[i]) for i in range(len(indices_used))]
    interpolated_data = interpolate_dataset(data, extend = 0, poly_order = 1)
    interpolated_ppw = torch.tensor([i[1] for i in interpolated_data], requires_grad = True)

    return interpolated_ppw, sample_mask



emissivity, sample_mask = gradient_per_image(depths1[0], periods[0])
torch.save(emissivity, f"default-emission.pt")
idx = sample_mask.nonzero(as_tuple=True)[0]
fom = ff.power_ratio(wavelengths[idx], emissivity[idx], ff.T_e, .726)
print(fom.detach())
