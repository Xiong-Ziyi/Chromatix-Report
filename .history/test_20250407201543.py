import jax.numpy as jnp

def forward_model(dmd, z):
    """
    Forward model for computer generated holography (CGH) simulation using digital micromirro device.
    
    Args:
        dmd (array): Digital micromirror device (DMD) pattern.
        z (array): distance to the sensor plane.

    Returns:
        array: Simulated Image.
    """

    