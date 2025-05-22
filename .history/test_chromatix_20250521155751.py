import jax
import jax.numpy as jnp
import optax
from chromatix.systems import OpticalSystem
from chromatix.elements import AmplitudeMask, PlaneWave
from chromatix import Field
from chromatix.functional import transfer_propagate
from skimage.data import cat
import matplotlib.pyplot as plt

# --- Parameters ---
H, W = 300, 300
z = 13e4
dx = 7.56
wavelength = 0.66
n = 1.0
N_pad = 150

# --- Prepare target image ---
img = cat().mean(axis=2)[:, 100:400]
target = jnp.array(img / img.max(), dtype=jnp.float32)
target = target.reshape(1, H, W, 1, 1)

# --- Initial DMD ---
key = jax.random.PRNGKey(4)
dmd = jax.nn.initializers.uniform(1.5)(key, (1, H, W, 1, 1)).astype(jnp.float32)

from chromatix import Field

def make_plane_field(shape, dx, spectrum):
    return Field(
        u=jnp.ones((1, *shape, 1, 1), dtype=jnp.complex64),
        dx=dx,
        spectrum=spectrum,
        spectral_density=1.0,
    )

# --- Define forward model ---
# --- Define forward model (no Flax modules) ---
def chrom_forward(dmd, z):
    field = make_plane_field((H, W), dx, wavelength)
    field = AmplitudeMask(amplitude=dmd.squeeze((0, 3, 4)), is_binary=True)(field)
    field = transfer_propagate(field, z=z, n=n, N_pad=N_pad, mode="same")
    return field.intensity

# --- Loss ---
def loss_fn(dmd, target, z):
    recon = chrom_forward(dmd, z)
    return optax.cosine_distance(recon.reshape(-1), target.reshape(-1)).mean()

# --- Optimizer ---
lr = 2.0
opt = optax.adam(lr)
opt_state = opt.init(dmd)

@jax.jit
def step(dmd, opt_state, target, z):
    loss, grads = jax.value_and_grad(loss_fn)(dmd, target, z)
    updates, opt_state = opt.update(grads, opt_state, dmd)
    dmd = optax.apply_updates(dmd, updates)
    return dmd, opt_state, loss

# --- Training loop ---
num_steps = 400
for i in range(num_steps):
    dmd, opt_state, loss = step(dmd, opt_state, target, z)
    if i % 40 == 0 or i == num_steps - 1:
        recon = chrom_forward((dmd > 0.5).astype(jnp.float32), z)
        corr = jnp.sum(recon * target) / (jnp.sqrt(jnp.sum(recon**2) * jnp.sum(target**2)) + 1e-8)
        print(f"[{i}] Loss: {loss.item():.4f}, Corr: {corr.item():.4f}")

# --- Final recon and DMD ---
final_dmd = (dmd > 0.5).astype(jnp.float32).squeeze()
recon = chrom_forward((dmd > 0.5).astype(jnp.float32), z).squeeze()

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recon, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(final_dmd, cmap="gray")
plt.title("DMD Pattern")
plt.axis("off")
plt.show()
