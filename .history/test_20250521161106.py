import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import cat

# --- Force JAX to use float32, like Chromatix ---
jax.config.update("jax_enable_x64", False)

# --- FFT wrappers ---
def fft(x, shift=False):
    axes = (-2, -1)
    if shift:
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    return jnp.fft.fft2(x, axes=axes)

def ifft(x, shift=False):
    axes = (-2, -1)
    if shift:
        return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    return jnp.fft.ifft2(x, axes=axes)

# --- STE binarization ---
@jax.custom_vjp
def ste_binarize(x):
    return (x > 0.5).astype(jnp.float32)

def ste_binarize_fwd(x):
    return (x > 0.5).astype(jnp.float32), ()

def ste_binarize_bwd(_, g):
    return (g,)

ste_binarize.defvjp(ste_binarize_fwd, ste_binarize_bwd)

# --- Forward model with padding, cropping, and phase kernel ---
def forward_model(dmd, z, wavelength=0.66, dx=7.56, n=1.0, N_pad=0):
    # Input shape: (1, H, W, 1, 1)
    dmd_2d = dmd.squeeze((0, 3, 4)).astype(jnp.float32)  # shape: (H, W)
    H, W = dmd_2d.shape

    # Binarize then pad
    bin_dmd = ste_binarize(dmd_2d)
    pad_width = ((N_pad, N_pad), (N_pad, N_pad))
    padded_dmd = jnp.pad(bin_dmd, pad_width, mode="constant", constant_values=0)

    # Propagation kernel
    H_pad, W_pad = padded_dmd.shape
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(H_pad, dx))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(W_pad, dx))
    kx, ky = jnp.meshgrid(kx, ky, indexing='ij')
    phase = -jnp.pi * (wavelength / n) * z * (kx**2 + ky**2)
    H_kernel = jnp.fft.ifftshift(jnp.exp(1j * phase)).astype(jnp.complex64)

    # FFT propagation (normalized)
    padded_dmd = padded_dmd.astype(jnp.complex64)
    field = ifft(fft(padded_dmd) * H_kernel)
    intensity = jnp.abs(field)**2

    # Crop back to original shape
    recon = intensity[N_pad:N_pad+H, N_pad:N_pad+W]

    return recon[None, ..., None, None]  # (1, H, W, 1, 1)

# --- Loss ---
def loss_fn(params, target, z):
    pred = forward_model(params, z)
    return optax.cosine_distance(pred.reshape(-1), target.reshape(-1)).mean()

grad_fn = jax.jit(jax.grad(loss_fn))

# --- Training ---
def train(init_params, target, z, steps=400, lr=2.0):
    opt = optax.adam(lr)
    opt_state = opt.init(init_params)
    params = init_params

    for i in range(steps):
        grads = grad_fn(params, target, z)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % 40 == 0 or i == steps - 1:
            recon_img = forward_model(params, z).squeeze()
            target_img = target.squeeze()
            loss = loss_fn(params, target, z)
            corr = jnp.sum(recon_img * target_img) / (
                jnp.sqrt(jnp.sum(recon_img**2) * jnp.sum(target_img**2)) + 1e-8
            )
            print(f"[{i}] Loss: {loss.item():.4f}, Corr: {corr.item():.4f}")

    return params

# --- Load and prepare target image ---
img = cat().mean(axis=2)[:, 100:400]
img = img / img.max()  # Normalize
H, W = 300, 300
target = jnp.array(img, dtype=jnp.float32).reshape(1, H, W, 1, 1)

# --- Initialization ---
key = jax.random.PRNGKey(4)
init = jax.nn.initializers.uniform(1.5)(key, (1, H, W, 1, 1)).astype(jnp.float32)

# --- Propagation distance ---
z = 13e4

# --- Run training ---
params = train(init, target, z)

# --- Final hard binarization and evaluation ---
binary_dmd = (params > 0.5).astype(jnp.float32)
recon = forward_model(binary_dmd, z).squeeze()
binary_dmd_2d = binary_dmd.squeeze()

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recon, cmap='gray')
plt.axis('off')
plt.title("Reconstructed Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_dmd_2d, cmap='gray')
plt.axis('off')
plt.title("DMD Pattern")

plt.show()
