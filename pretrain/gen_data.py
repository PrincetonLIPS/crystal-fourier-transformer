import jax
import jax.numpy as jnp
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple

MIN_LATTICE_LENGTH = 0.8  # Angstroms
MAX_LATTICE_LENGTH = 16.0  # Angstroms
NUM_SAMPLES_PER_SPACE_GROUP = 125000
BATCH_SIZE = 25000 

def get_bravais_lattice_type(space_group):
    if 1 <= space_group <= 2:
        return "triclinic"
    elif 3 <= space_group <= 15:
        return "monoclinic"
    elif 16 <= space_group <= 74:
        return "orthorhombic"
    elif 75 <= space_group <= 142:
        return "tetragonal"
    elif 143 <= space_group <= 194:
        return "hexagonal"
    elif 195 <= space_group <= 230:
        return "cubic"
    else:
        raise ValueError(f"Invalid space group: {space_group}")

def generate_lattice_vectors(key, space_group):
    """Generate lattice vectors adhering to the Bravais lattice type."""
    key1, key2 = jax.random.split(key, 2)
    lattice_type = get_bravais_lattice_type(space_group)
    
    def sample_length(key):
        length = jax.random.uniform(key, (), minval=MIN_LATTICE_LENGTH, maxval=MAX_LATTICE_LENGTH)
        return length
    
    def sample_angle(key):
        return jax.random.uniform(key, (), minval=jnp.pi/8, maxval=jnp.pi*7/8)
    
    if lattice_type == "triclinic":
        a, b, c = jax.vmap(sample_length)(jax.random.split(key1, 3))
        alpha, beta, gamma = jax.vmap(sample_angle)(jax.random.split(key2, 3))
        # First vector along x-axis
        ax, ay, az = a, 0.0, 0.0
        # Second vector in x-y plane
        bx, by, bz = b * jnp.cos(gamma), b * jnp.sin(gamma), 0.0
        # Third vector
        cx, cy, cz = c * jnp.cos(beta), c * jnp.sin(beta) * jnp.cos(alpha), c * jnp.sin(beta) * jnp.sin(alpha)
        volume = ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx) + az*(bx*cy - by*cx)
        
        lattice_vectors = jnp.array([
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz]
        ])
    elif lattice_type == "monoclinic":
        a, b, c = jax.vmap(sample_length)(jax.random.split(key1, 3))
        beta = jax.random.uniform(key1, (), minval=jnp.pi/6, maxval=jnp.pi*5/6)
        lattice_vectors = jnp.array([
            [a, 0, 0],
            [0, b, 0],
            [c * jnp.cos(beta), 0, c * jnp.sin(beta)]
        ])
    elif lattice_type == "orthorhombic":
        lattice_vectors = jnp.diag(jax.vmap(sample_length)(jax.random.split(key2, 3)))
    elif lattice_type == "tetragonal":
        a = sample_length(key1)
        c = sample_length(key2)
        lattice_vectors = jnp.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
    elif lattice_type == "hexagonal":
        a = sample_length(key1)
        c = sample_length(key2)
        lattice_vectors = jnp.array([[a, 0, 0], [-a/2, a*jnp.sqrt(3)/2, 0], [0, 0, c]])
    elif lattice_type == "cubic":
        a = sample_length(key2)
        lattice_vectors = a * jnp.eye(3)
    return lattice_vectors

def generate_synthetic_sample(key: jnp.ndarray, space_group: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a crystal structure consisting of 2 atoms."""
    key1, key2 = jax.random.split(key, 2)
    lattice_vectors = generate_lattice_vectors(key1, space_group)
    general_pos = jax.random.uniform(key2, (2, 3))
    return general_pos, lattice_vectors

def generate_batch(key, num_samples, space_group):
    keys = jax.random.split(key, num_samples)
    return jax.vmap(lambda k: generate_synthetic_sample(k, space_group))(keys)

def save_data(positions, lattice_vectors, space_group, output_dir):
    filename = os.path.join(output_dir, f"space_group_{space_group}.npz")
    np.savez_compressed(
        filename,
        positions=np.array(positions),
        lattice_vectors=np.array(lattice_vectors),
        space_group=np.full(positions.shape[0], space_group)
    )

def main():
    output_dir = "data/synthetic_crystals_hex"
    os.makedirs(output_dir, exist_ok=True)
    key = jax.random.PRNGKey(42)

    hex_space_groups = list(range(143, 195)) 
    cubic_space_groups = list(range(1, 143)) + list(range(195, 231))

    for space_group in hex_space_groups:
        all_positions = []
        all_lattice_vectors = []

        for _ in range(0, NUM_SAMPLES_PER_SPACE_GROUP, BATCH_SIZE):
            key, subkey = jax.random.split(key)
            positions, lattice_vectors = generate_batch(subkey, BATCH_SIZE, space_group)
            
            all_positions.append(positions)
            all_lattice_vectors.append(lattice_vectors)

        positions = jnp.concatenate(all_positions)
        lattice_vectors = jnp.concatenate(all_lattice_vectors)

        save_data(positions, lattice_vectors, space_group, output_dir)

    print(f"Data generation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    main()