import jax.numpy as jnp
from chromatix.utils import fft, ifft
import optax

def forward_model(dmd: jnp.ndarray,
                    z: jnp.ndarray,
                    wavelength: jnp.ndarray = 0.66,
                    dx: jnp.ndarray = 1.0,
                    n: jnp.ndarray = 1.0) -> jnp.ndarray:
    """
    Forward model for computer-generated holography (CGH) simulation using a digital micromirror device (DMD).
    """
    k_grid = jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0])),
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1])),
    ) / dx

    phase = -jnp.pi * (wavelength / n) * z * jnp.sum(k_grid**2, axis=0)
    transfer_fn = jnp.fft.ifftshift(jnp.exp(1j * phase))

    dmd_amp = ifft(fft(dmd) * transfer_fn)
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
