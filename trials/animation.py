#!/usr/bin/env python3
import torch, ff
import matplotlib
# pick a real GUI backend; if you get "Invalid DISPLAY" errors,
# try 'Agg'→no GUI, or install PyQt5 then use 'Qt5Agg'.
matplotlib.use('Qt5Agg')  

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1) build your data
wls = torch.linspace(0.35, 3.0, 2651)
eps = torch.zeros_like(wls, requires_grad=True)
with torch.no_grad():
    eps[(wls >= 1.0) & (wls < 2.0)] = 0.7
    eps[(wls >= 2.0)]                    = 0.3

# 2) make the figure
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(wls.numpy(), eps.detach().numpy(), lw=2)
ax.set_xlim(0.35, 3.0)
ax.set_ylim(0, 1)
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Emissivity')

# 3) define your update step
def update(i):
    # zero‐grad, compute FOM, backprop, step & clamp
    if eps.grad is not None:
        eps.grad.zero_()
    fom = ff.power_ratio(wls, eps, ff.T_e, .726)
    fom.backward()
    with torch.no_grad():
        # you can scale the step by lr if this is too jumpy
        eps.add_(eps.grad)
        eps.clamp_(0,1)
    line.set_ydata(eps.detach().numpy())
    assert eps.max()<=1
    ax.set_title(f"Iter {i+1}, FOM={fom.item():.4f}")
    return (line,)

# 4) create the animation (turn blit off if it misbehaves)
anim = FuncAnimation(fig, update, frames=100, interval=1, blit=False)

# 5) hand control over to the GUI — *this* actually runs your animation loop
plt.show()
