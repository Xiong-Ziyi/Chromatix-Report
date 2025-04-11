import jax.numpy as jnp
from functools import partial

def forward_model(dmd: jnp.ndarray,
                    z: jnp.ndarray,
                    wavelength: jnp.ndarray = 0.66,
                    dx: jnp.ndarray = 1.0,
                    n: jnp.ndarray = 1.0) -> jnp.ndarray:
    """
    Forward model for computer-generated holography (CGH) simulation using a digital micromirror device (DMD).
    
    Args:
        dmd (array): Digital micromirror device (DMD) pattern.
        z (array): distance to the sensor plane.

    Returns:
        array: Simulated Image.
    """
    k_grid = jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[0])),
        jnp.fft.fftshift(jnp.fft.fftfreq(dmd.shape[1])),
    ) / dx

    phase = -jnp.pi * (wavelength / n) * z * jnp.sum(k_grid**2, axis=0)
    transfer_fn = jnp.fft.ifftshift(jnp.exp(1j * phase))

    


