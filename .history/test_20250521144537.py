import jax
import jax.numpy as jnp
import optax
import numpy as np
from skimage.data import cat
import matplotlib.pyplot as plt

# --- FFT wrappers ---
def fft(x, axes=(-3, -2), shift=False):
    if shift:
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    return jnp.fft.fft2(x, axes=axes)

def ifft(x, axes=(-3, -2), shift=False):
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

# --- Forward model ---
def forward_model(dmd, z, wavelength=0.66, dx=7.56, n=1.0):
    # dmd shape: (1, H, W, 1, 1)
    dmd_2d = dmd.squeeze((0, 3, 4))  # shape: (H, W)

    kx = jnp.fft.fftshift(jnp.fft.fftfreq(dmd_2d.shape[0], dx))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(dmd_2d.shape[1], dx))
    kx, ky = jnp.meshgrid(kx, ky, indexing='ij')
    phase = -jnp.pi * (wavelength / n) * z * (kx**2 + ky**2)
    H = jnp.fft.ifftshift(jnp.exp(1j * phase))

    bin_dmd = ste_binarize(dmd_2d)
    field = ifft(fft(bin_dmd) * H)
    intensity = jnp.abs(field)**2
    return intensity[None, ..., None, None]  # shape: (1, H, W, 1, 1)

# --- Loss function ---
def loss_fn(params, target, z):
    pred = forward_model(params, z)
    return optax.cosine_distance(pred.reshape(-1), target.reshape(-1)).mean()

# --- Gradient function ---
grad_fn = jax.jit(jax.grad(loss_fn))

# --- Training loop ---
def train(init_params, target, z, steps=400, lr=2.0):
    opt = optax.adam(lr)
    opt_state = opt.init(init_params)
    params = init_params

    for i in range(steps):
        grads = grad_fn(params, target, z)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % 40 == 0 or i == steps - 1:
            recon = forward_model(params, z).squeeze()
            loss = loss_fn(params, target, z)
            corr = jnp.sum(recon * target) / (jnp.sqrt(jnp.sum(recon**2) * jnp.sum(target**2)) + 1e-8)
            print(f"[{i}] Loss: {loss:.4f}, Corr: {corr:.4f}")

    return params

# --- Main ---
img = cat().mean(axis=2)[:, 100:400]
target = jnp.array(img / img.max())  # normalize to [0, 1]
target = target[None, ..., None, None]  # (1, H, W, 1, 1)

H, W = 300, 300
key = jax.random.PRNGKey(4)
init = jax.nn.initializers.uniform(1.5)(key, (1, H, W, 1, 1))

z = 13e4
params = train(init, target, z)

# --- Final evaluation ---
binary_dmd = (params > 0.5).astype(jnp.float32)
recon = forward_model(binary_dmd, z).squeeze()

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recon, cmap='gray')
plt.axis('off')
plt.title("Reconstructed Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_dmd.squeeze(), cmap='gray')
plt.axis('off')
plt.title("DMD Pattern")

plt.show()
