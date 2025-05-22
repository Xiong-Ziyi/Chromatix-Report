import jax
import jax.numpy as jnp
import optax
import numpy as np
from skimage.data import cat
from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt

# --- FFT wrappers ---
def fft(x, axes=(0, 1), shift=False):
    if shift:
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x), axes=axes), axes=axes)
    return jnp.fft.fft2(x, axes=axes)

def ifft(x, axes=(0, 1), shift=False):
    if shift:
        return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x), axes=axes), axes=axes)
    return jnp.fft.ifft2(x, axes=axes)

# --- Surrogate binarization ---
@jax.custom_vjp
def ste_binarize(x):
    return (x > 0.5).astype(jnp.float32)

def ste_binarize_fwd(x):
    y = (x > 0.5).astype(jnp.float32)
    return y, ()

def ste_binarize_bwd(_, g):
    return (g,)  # Constant 1.0 gradient like STE

ste_binarize.defvjp(ste_binarize_fwd, ste_binarize_bwd)


# --- Forward model ---
def forward_model(dmd, z, wavelength=0.66, dx=7.56, n=1.0):
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0], dx))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1], dx))
    kx, ky = jnp.meshgrid(kx, ky, indexing='ij')
    phase = -jnp.pi * (wavelength / n) * z * (kx**2 + ky**2)
    H = jnp.fft.ifftshift(jnp.exp(1j * phase))

    bin_dmd = binarize_with_surrogate(dmd)
    field = ifft(fft(bin_dmd) * H)
    return jnp.abs(field)**2

# --- Loss ---
def loss_fn(params, target, z):
    recon = forward_model(params, z)
    return optax.cosine_distance(recon.flatten(), target.flatten()).mean()

# --- Gradient ---
grad_fn = jax.jit(jax.grad(loss_fn))

# --- Training ---
def train_cgh(init_params, target, z, lr=2.0, steps=400):
    opt = optax.adam(lr)
    opt_state = opt.init(init_params)
    params = init_params

    history = {"loss": [], "correlation": []}

    for i in range(steps):
        grads = grad_fn(params, target, z)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % 40 == 0 or i == steps - 1:
            recon = forward_model(params, z)
            loss = loss_fn(params, target, z)
            corr = jnp.sum(recon * target) / (jnp.sqrt(jnp.sum(recon**2) * jnp.sum(target**2)) + 1e-8)
            history["loss"].append(loss.item())
            history["correlation"].append(corr.item())
            print(f"[{i}] loss: {loss:.4f}, corr: {corr:.4f}")

    return params, history

# --- Main Execution ---
img = cat().mean(axis=2)
target = jnp.array(img[:, 100:400])

key = jax.random.PRNGKey(4)
init = jax.nn.initializers.uniform(1.5)(key, (300, 300))
z = 13e4

trained_dmd, hist = train_cgh(init, target, z)

final_dmd = (trained_dmd > 0.5).astype(jnp.float32)
recon = forward_model(final_dmd, z)

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recon, cmap='gray')
plt.axis('off')
plt.title("Reconstructed Image")

plt.subplot(1, 2, 2)
plt.imshow(final_dmd, cmap='gray')
plt.axis('off')
plt.title("DMD Pattern")

plt.show()
