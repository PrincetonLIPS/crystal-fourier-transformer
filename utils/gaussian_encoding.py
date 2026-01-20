import jax
import jax.numpy as jnp
from typing import Tuple
from pretrain.mlp import ALL_ROTATIONS, ALL_TRANSLATIONS, ALL_OP_COUNTS

def _get_space_group_ops_fractional(space_group_num: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Gather precomputed (A_f, t_f, count) from global arrays for the given space group.

    Returns:
        A_f: (M, 3, 3), t_f: (M, 3), count: () int32 number of valid ops
    """
    idx = jnp.asarray(space_group_num, dtype=jnp.int32) - 1
    A_f = ALL_ROTATIONS[idx]
    t_f = ALL_TRANSLATIONS[idx]
    count = ALL_OP_COUNTS[idx]
    return A_f.astype(jnp.float32), t_f.astype(jnp.float32), count.astype(jnp.int32)


def select_k_vectors(num_coefficients: int) -> jnp.ndarray:
    """
    Select num_coefficients integer k-vectors in Z^3 by increasing Euclidean norm,
    tie-broken lexicographically. Returns shape (num_coefficients, 3), dtype=int32.
    Always excludes k = (0, 0, 0).
    """
    # Grow radius until we collect enough k points
    collected = []
    radius = 0
    while len(collected) < num_coefficients:
        radius += 1
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    vec = (i, j, k)
                    # Skip the origin explicitly
                    if vec == (0, 0, 0):
                        continue
                    # Skip those outside the shell just added
                    if max(abs(i), abs(j), abs(k)) != radius and radius > 1:
                        continue
                    collected.append(vec)

        # Sort all seen so far by norm then lexicographic and trim
        collected = sorted(collected, key=lambda v: (v[0] * v[0] + v[1] * v[1] + v[2] * v[2], v))
        collected = collected[: num_coefficients]

    return jnp.array(collected, dtype=jnp.int32)


def select_k_vectors_full_shell(min_num_coefficients: int) -> Tuple[jnp.ndarray, int, int]:
    """
    Select ALL integer k-vectors in Z^3 within the smallest Euclidean radius r
    that contains at least min_num_coefficients non-zero k-points.

    Returns:
        k_vectors: (K_actual, 3) int32 array of all k with ||k||_2 <= r, k != 0
        radius_used: int radius r used
        K_actual: int number of returned k-vectors
    """
    if min_num_coefficients <= 0:
        return jnp.empty((0, 3), dtype=jnp.int32), 0, 0

    r = 0.0
    K_actual = 0
    k_list = []
    while K_actual < min_num_coefficients:
        r += 0.2
        k_list = []
        r2 = r * r
        for i in range(-int(r), int(r) + 1):
            for j in range(-int(r), int(r) + 1):
                for k in range(-int(r), int(r) + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if (i * i + j * j + k * k) <= r2:
                        k_list.append((i, j, k))
        # Sort by norm then lexicographic for determinism
        k_list = sorted(k_list, key=lambda v: (v[0] * v[0] + v[1] * v[1] + v[2] * v[2], v))
        K_actual = len(k_list)

    k_vectors = jnp.array(k_list, dtype=jnp.int32)
    return k_vectors, int(jnp.ceil(r)), K_actual


def fourier_coefficients_gaussian_orbit(
    x_frac: jnp.ndarray,
    B: jnp.ndarray,
    space_group_num: int,
    k_vectors_frac: jnp.ndarray,
    sigma: float,
    batch_size: int = 128,
) -> jnp.ndarray:
    """
    Compute complex Fourier coefficients for the Gaussian orbit density centered at x.

    Args:
        x_frac: (3,) fractional coordinate of the atom in the crystal's lattice basis B
        B: (3, 3) lattice matrix with columns as lattice vectors in Cartesian coords
        space_group_num: international number 1..230
        k_vectors_frac: (K, 3) integer k-vectors in fractional reciprocal coordinates
        sigma: Gaussian width in Cartesian units (same units as columns of B)
        batch_size: number of k-vectors to process at once to manage memory usage

    Returns:
        coeffs: (K,) complex64 array of Fourier coefficients
    """
    A_f, t_f, op_count = _get_space_group_ops_fractional(space_group_num)

    # Convert to Cartesian coordinates
    B_inv = jnp.linalg.inv(B)
    x_cart = B @ x_frac  # (3,)
    
    # Transform operations to Cartesian once
    A_c = B @ A_f @ B_inv  # (M, 3, 3)
    t_c = (B @ t_f[..., None])[..., 0]  # (M, 3)
    
    # Precompute Ax + t for all symmetry operations
    Ax_plus_t = (A_c @ x_cart) + t_c  # (M, 3)
    
    # Create operation mask once
    op_mask = (jnp.arange(A_c.shape[0]) < op_count).astype(jnp.float32)

    # Process k-vectors in batches
    num_k = k_vectors_frac.shape[0]
    coeffs_list = []
    
    for start_idx in range(0, num_k, batch_size):
        end_idx = min(start_idx + batch_size, num_k)
        k_batch = k_vectors_frac[start_idx:end_idx]
        
        # Convert k-vectors to Cartesian for this batch
        k_cart_batch = (B_inv.T @ k_batch.T).T  # (batch, 3)
        
        # Compute decay factors for this batch
        k_norm_sq = jnp.sum(k_cart_batch * k_cart_batch, axis=-1)  # (batch,)
        decay = jnp.exp(-2.0 * (jnp.pi ** 2) * (sigma ** 2) * k_norm_sq)  # (batch,)
        
        # Compute phases and contributions for all operations and k-vectors in batch
        # Shape: (batch, M, 1) @ (M, 3) -> (batch, M)
        phases = -2.0 * jnp.pi * 1j * (k_cart_batch @ Ax_plus_t.T)  # (batch, M)
        contribs = jnp.exp(phases) * op_mask  # (batch, M)
        
        # Sum over symmetry operations
        summed = jnp.sum(contribs, axis=1)  # (batch,)
        coeffs_batch = decay * summed
        
        coeffs_list.append(coeffs_batch)
    
    # Concatenate all batches
    coeffs = jnp.concatenate(coeffs_list)
    return coeffs.astype(jnp.complex64)


def encode_atom_gaussian_fourier(
    x_frac: jnp.ndarray,
    B: jnp.ndarray,
    space_group_num: int,
    k_vectors_frac: jnp.ndarray,
    sigma: float,
    batch_size: int = 128,
) -> jnp.ndarray:
    """
    Real-valued encoding vector of length 2*K formed by [real, imag] parts of coefficients.
    """
    coeffs = fourier_coefficients_gaussian_orbit(x_frac, B, space_group_num, k_vectors_frac, sigma, batch_size=batch_size)
    real = jnp.real(coeffs)
    imag = jnp.imag(coeffs)
    return jnp.concatenate([real, imag], axis=0)

def compute_batch_encodings(
    atom_positions_frac: jnp.ndarray,
    lattice_matrices: jnp.ndarray,
    space_groups: jnp.ndarray,
    masks: jnp.ndarray,
    embedding_dim: int,
    sigma: float,
    k_vectors_frac: jnp.ndarray = None,
    k_batch_size: int = 128,
    atom_batch_size: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Compute Gaussian Fourier encodings for a batch of crystals.

    Args:
        atom_positions_frac: (B, N, 3)
        lattice_matrices: (B, 3, 3)
        space_groups: (B,)
        masks: (B, N) boolean/0-1
        embedding_dim: output dim per atom (must be even)
        sigma: Gaussian width in Cartesian units
        k_vectors_frac: optional (K, 3). If None, chosen globally with K=embedding_dim//2
        k_batch_size: number of k-vectors to process at once to manage memory usage

    Returns:
        encodings: (B, N, D_actual)
        k_vectors_frac: (K_actual, 3) used
        embedding_dim_actual: int, equals 2 * K_actual
    """
    # Determine k-vectors and actual embedding dim. If a k set is provided, honor it.
    if k_vectors_frac is None:
        # Expand to full Euclidean shell that reaches at least K_desired
        K_desired = max(1, embedding_dim // 2)
        k_vectors_frac, _r, K_actual = select_k_vectors_full_shell(K_desired)
    else:
        K_actual = int(k_vectors_frac.shape[0])
    D_actual = 2 * K_actual

    def encode_single_crystal(positions, B, sg, mask):
        # Process atoms in chunks to keep peak memory low
        num_atoms = positions.shape[0]
        chunks = []
        for start in range(0, num_atoms, atom_batch_size):
            end = min(start + atom_batch_size, num_atoms)
            pos_chunk = positions[start:end]
            def enc_single_atom(xf):
                return encode_atom_gaussian_fourier(xf, B, sg, k_vectors_frac, sigma, batch_size=k_batch_size)
            enc_chunk = jax.vmap(enc_single_atom)(pos_chunk)  # (chunk, D)
            chunks.append(enc_chunk)
        enc_all = jnp.concatenate(chunks, axis=0)
        enc_all = enc_all * mask[:, None]
        return enc_all

    # Compute each crystal sequentially to reduce peak memory
    outputs = []
    B_total = atom_positions_frac.shape[0]
    for b in range(B_total):
        enc_b = encode_single_crystal(
            atom_positions_frac[b],
            lattice_matrices[b],
            space_groups[b],
            masks[b].astype(jnp.float32),
        )
        outputs.append(enc_b)
    encodings = jnp.stack(outputs, axis=0)
    return encodings, k_vectors_frac, int(D_actual)


def reconstruct_density_from_coeffs(
    coeffs: jnp.ndarray,
    k_vectors_frac: jnp.ndarray,
    B: jnp.ndarray,
    y_cart: jnp.ndarray,
) -> jnp.ndarray:
    """
    Reconstruct rho(y) at Cartesian point y from Fourier coefficients of a single atom.
    coeffs are complex of shape (K,), k_vectors_frac shape (K, 3).
    """
    k_cart = (jnp.linalg.inv(B).T @ k_vectors_frac.T).T  # (K, 3)
    phases = 2.0 * jnp.pi * 1j * (k_cart @ y_cart)  # (K,)
    return jnp.real(jnp.sum(coeffs * jnp.exp(phases)))
