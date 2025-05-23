import jax
import jax.numpy as jnp
import optax
from skimage.data import cat
from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt

def fft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    """
    Computes ``fft2`` for input of shape `(B... H W C)`.
    If shift is true, first applies ``ifftshift``, than an ``fftshift`` to
    make sure everything stays centered.
    """
    fft = partial(jnp.fft.fft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(fft(ifftshift(x)))
    else:
        return fft(x)

def ifft(x: jnp.ndarray, axes: Tuple[int, int] = (0, 1), shift: bool = False) -> jnp.ndarray:
    """
    Computes ``ifft2`` for input of shape `(B... H W C)`.
    If shift is true, first applies ``ifftshift``, than an ``fftshift`` to
    make sure everything stays centered.
    """
    ifft = partial(jnp.fft.ifft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(ifft(ifftshift(x)))
    else:
        return ifft(x)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the sigmoid function.
    """
    return 1 / (1 + jnp.exp(-37 * (x - 0.5)))

def forward_model(dmd: jnp.ndarray,
                    z: jnp.ndarray,
                    wavelength: jnp.ndarray = 0.66,
                    dx: jnp.ndarray = 1.0,
                    n: jnp.ndarray = 1.0) -> jnp.ndarray:
    """
    Forward model for computer-generated holography (CGH) simulation using a digital micromirror device (DMD).
    """
    k_grid = jnp.array(jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0])),
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1])),
        indexing = "ij",
    )) / dx

    phase = -jnp.pi * (wavelength / n) * z * jnp.sum(k_grid**2, axis=0)
    transfer_fn = jnp.fft.ifftshift(jnp.exp(1j * phase))

    binarized_dmd = dmd # sigmoid(dmd)
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
    Loss function for the forward model.
    """
    img = forward_model(dmd, z, wavelength, dx, n)

    loss = optax.cosine_distance(
        img.reshape(-1), target.reshape(-1), epsilon =1e-8
    ).mean()

    return loss

grad_fn = jax.jit(jax.grad(loss_fn))

def adam_optimizer(dmd: jnp.ndarray, 
                   target: jnp.ndarray, 
                   z: jnp.ndarray,
                   num_iterations) -> jnp.ndarray:
    """
    Adam optimizer for the forward model.
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

img = cat().mean(2)
img = img[:, 100:400]
target = jnp.array(img)

num_iterations = 500
z = 13e4
dmd = jax.random.uniform(jax.random.PRNGKey(0), shape=(300, 300))

dmd = adam_optimizer(dmd, target, z, num_iterations)

dmd = (dmd > 0.5).astype(jnp.float32) * 1.0
img = forward_model(dmd, z)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Reconstructed Image')   

plt.subplot(1, 2, 2)
plt.imshow(dmd, cmap='gray')
plt.axis('off')
plt.title('DMD Pattern')

plt.show()