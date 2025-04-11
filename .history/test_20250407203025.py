import jax.numpy as jnp

def forward_model(dmd: jnp.ndarray,
                    z: jnp.ndarray,
                    wavelength: jnp.ndarray = 0.66,
                    dx: jnp.ndarray = 1.0) -> jnp.ndarray:
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

    

    
    transfer_fn = jnp.exp()
