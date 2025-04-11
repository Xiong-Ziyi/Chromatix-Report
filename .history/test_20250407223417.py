import jax
import jax.numpy as jnp
import optax
from skimage.data import cat
from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt

# Custom binarization with a straight-through estimator.
@jax.custom_gradient
def binarize(x: jnp.ndarray) -> jnp.ndarray:
    # Forward pass: hard thresholding at 0.5.
    y = (x > 0.5).astype(x.dtype)
    def grad(dy):
        # Surrogate gradient: derivative of a steep sigmoid.
        sigmoid_val = 1 / (1 + jnp.exp(-37 * (x - 0.5)))
        grad_val = 37 * sigmoid_val * (1 - sigmoid_val)
        return dy * grad_val
    return y, grad

def fft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    fft_fn = partial(jnp.fft.fft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(fft_fn(ifftshift(x)))
    else:
        return fft_fn(x)

def ifft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    ifft_fn = partial(jnp.fft.ifft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(ifft_fn(ifftshift(x)))
    else:
        return ifft_fn(x)

def forward_model(dmd: jnp.ndarray,
                  z: jnp.ndarray,
                  wavelength: jnp.ndarray = 0.66,
                  dx: jnp.ndarray = 1.0,
                  n: jnp.ndarray = 1.0) -> jnp.ndarray:
    """
    Forward model for CGH simulation.
    """
    k_grid = jnp.array(jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0])),
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1])),
        indexing="ij",
    )) / dx

    phase = -jnp.pi * (wavelength / n) * z * jnp.sum(k_grid**2, axis=0)
    transfer_fn = jnp.fft.ifftshift(jnp.exp(1j * phase))
    
    # Use the custom binarize function for differentiable binarization.
    binarized_dmd = binarize(dmd)
    
    dmd_amp = ifft(fft(binarized_dmd) * transfer_fn)
    img = jnp.abs(dmd_amp)**2
    return img

def loss_fn(dmd: jnp.ndarray,
            target: jnp.ndarray,
            z: jnp.ndarray,
            wavelength: jnp.ndarray = 0.66,
            dx: jnp.ndarray = 1.0,
            n: jnp.ndarray = 1.0) -> jnp.ndarray:
    """
    Loss function based on cosine distance between the propagated image and target.
    """
    img = forward_model(dmd, z, wavelength, dx, n)
    loss = optax.cosine_distance(
        img.reshape(-1), target.reshape(-1), epsilon=1e-8
    ).mean()
    return loss

grad_fn = jax.jit(jax.grad(loss_fn))

def adam_optimizer(dmd: jnp.ndarray, 
                   target: jnp.ndarray, 
                   z: jnp.ndarray,
                   num_iterations: int) -> jnp.ndarray:
    """
    Adam optimizer for updating the DMD pattern.
    """
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

# Load and preprocess target image.
img = cat().mean(2)
img = img[:, 100:400]
target = jnp.array(img)

num_iterations = 500
z = 13e4
dmd = jax.random.uniform(jax.random.PRNGKey(0), shape=(300, 300))

# Optimize the DMD pattern using the STE-based binarization.
dmd = adam_optimizer(dmd, target, z, num_iterations)

# After optimization, apply a hard threshold for the final output.
dmd_final = (dmd > 0.5).astype(jnp.float32)
img_recon = forward_model(dmd_final, z)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_recon, cmap='gray')
plt.axis('off')
plt.title('Reconstructed Image')   

plt.subplot(1, 2, 2)
plt.imshow(dmd_final, cmap='gray')
plt.axis('off')
plt.title('DMD Pattern')

plt.show()
