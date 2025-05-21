import jax
import jax.numpy as jnp
import optax
from skimage.data import cat
from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt

# --- FFT and iFFT utils (unchanged) ---
def fft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    fft = partial(jnp.fft.fft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(fft(ifftshift(x)))
    else:
        return fft(x)

def ifft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    ifft = partial(jnp.fft.ifft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(ifft(ifftshift(x)))
    else:
        return ifft(x)

# --- Define surrogate binarization ---
@jax.custom_vjp
def binarize_with_surrogate(x: jnp.ndarray) -> jnp.ndarray:
    return (x > 0.5).astype(jnp.float32)

def binarize_with_surrogate_fwd(x):
    y = (x > 0.5).astype(jnp.float32)
    return y, (x,)  # <-- pack into a tuple!

def binarize_with_surrogate_bwd(res, g):
    (x,) = res  # <-- unpack from tuple
    surrogate_grad = 37 * jnp.exp(-37 * (x - 0.5)) / ((1 + jnp.exp(-37 * (x - 0.5)))**2)
    return (g * surrogate_grad,)


binarize_with_surrogate.defvjp(binarize_with_surrogate_fwd, binarize_with_surrogate_bwd)

# --- Forward model ---
def forward_model(dmd: jnp.ndarray, z: jnp.ndarray, wavelength: jnp.ndarray = 0.66, dx: jnp.ndarray = 1.0, n: jnp.ndarray = 1.0) -> jnp.ndarray:
    k_grid = jnp.array(jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0])),
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1])),
        indexing="ij",
    )) / dx
    phase = -jnp.pi * (wavelength / n) * z * jnp.sum(k_grid**2, axis=0)
    transfer_fn = jnp.fft.ifftshift(jnp.exp(1j * phase))

    binarized_dmd = binarize_with_surrogate(dmd)
    dmd_amp = ifft(fft(binarized_dmd) * transfer_fn)
    img = jnp.abs(dmd_amp)**2
    return img

# --- Loss ---
def loss_fn(dmd: jnp.ndarray, target: jnp.ndarray, z: jnp.ndarray, wavelength: jnp.ndarray = 0.66, dx: jnp.ndarray = 1.0, n: jnp.ndarray = 1.0) -> jnp.ndarray:
    img = forward_model(dmd, z, wavelength, dx, n)
    loss = optax.cosine_distance(img.reshape(-1), target.reshape(-1), epsilon=1e-8).mean()
    return loss

# --- Training ---
grad_fn = jax.jit(jax.grad(loss_fn))

def adam_optimizer(dmd: jnp.ndarray, target: jnp.ndarray, z: jnp.ndarray, num_iterations) -> jnp.ndarray:
    opt_init, opt_update = optax.adam(learning_rate=0.1)
    opt_state = opt_init(dmd)

    for i in range(num_iterations):
        grads = grad_fn(dmd, target, z)
        updates, opt_state = opt_update(grads, opt_state, dmd)
        dmd = optax.apply_updates(dmd, updates)

        if i % 50 == 0:
            loss = loss_fn(dmd, target, z)
            print(f"Iteration {i}, Loss: {loss}")

    return dmd

# --- Main execution ---
img = cat().mean(2)
img = img[:, 100:400]
target = jnp.array(img)

num_iterations = 500
z = 13e4
dmd = jax.random.uniform(jax.random.PRNGKey(0), shape=(300, 300))

dmd = adam_optimizer(dmd, target, z, num_iterations)

# Final hard binarization for visualization
final_dmd = (dmd > 0.5).astype(jnp.float32)
recon_img = forward_model(final_dmd, z)

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recon_img, cmap='gray')
plt.axis('off')
plt.title('Reconstructed Image')

plt.subplot(1, 2, 2)
plt.imshow(final_dmd, cmap='gray')
plt.axis('off')
plt.title('DMD Pattern')

plt.show()
