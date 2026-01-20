import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import os
import argparse
import pickle
from flax.training import train_state, checkpoints
from typing import Any
from tqdm import tqdm
from utils.space_graphs import SpaceGraph
from space_groups import SpaceGroup
from space_groups.utils import sympy_to_numpy

def parse_args():
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--l2_weight", type=float, default=5e-5, help="L2 weight for regularization")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for training")

    # Model params
    parser.add_argument("--fourier_dim", type=int, default=300, help="Dimension of the complex Fourier encoding")
    parser.add_argument("--res_block_dim", type=int, default=256, help="Dimension of the inputs/outputs of the residual blocks")
    parser.add_argument("--num_pos_res_blocks", type=int, default=3, help="Number of position residual blocks")
    parser.add_argument("--num_lattice_res_blocks", type=int, default=3, help="Number of lattice residual blocks")
    parser.add_argument("--initial_embedding_dim", type=int, default=512, help="Dimension of initial embedding")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of final embedding")

    # Paths and logging
    parser.add_argument("--data_dir", type=str, default="data/synthetic_crystals_hex", help="Directory containing synthetic crystal data")
    parser.add_argument("--version", type=str, default="default", help="Version of the model")
    parser.add_argument("--ckpt", type=str, default="trained_state", help="Path to pretrained model state")
    parser.add_argument("--crystal_system", type=str, default="hexagonal", choices=["cubic", "hexagonal"],
                        help="Crystal system to pretrain: 'cubic' or 'hexagonal'")
    args = parser.parse_args()
    return vars(args)

def precompute_space_group_operations():
    all_rotations = []
    all_translations = []
    all_bases = []
    all_op_counts = []

    for sg in range(1, 231):
        group = SpaceGroup(sg)
        operations = sympy_to_numpy(group.operations)
        basis = sympy_to_numpy(group.basic_basis)
        A_list = [basis @ op[:3, :3] @ np.linalg.inv(basis) for op in operations]
        t_list = [basis @ op[:3, 3] for op in operations]
        num_ops = len(A_list)  # Store the actual number of operations before padding
        padded_rotations = A_list + [np.eye(3)] * (192 - len(A_list))   # Pad to max number of operations
        padded_translations = t_list + [np.zeros(3)] * (192 - len(t_list))
        all_rotations.append(jnp.array(padded_rotations, dtype=jnp.float32))
        all_translations.append(jnp.array(padded_translations, dtype=jnp.float32))
        all_bases.append(jnp.array(basis, dtype=jnp.float32))
        all_op_counts.append(num_ops)

    return jnp.array(all_rotations), jnp.array(all_translations), jnp.array(all_bases), jnp.array(all_op_counts, dtype=jnp.int32)

ALL_ROTATIONS, ALL_TRANSLATIONS, ALL_BASES, ALL_OP_COUNTS = precompute_space_group_operations()

class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(features=self.features * 2)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        out = x + residual
        return out

class MLP(nn.Module):
    config: dict

    def setup(self):
        # Position branch
        self.pos_dense1 = nn.Dense(features=self.config['initial_embedding_dim'])
        self.pos_dense2 = nn.Dense(features=self.config['res_block_dim'])
        self.pos_res_blocks = [ResidualBlock(features=self.config['res_block_dim']) 
                             for _ in range(self.config['num_pos_res_blocks'])]
        
        # Lattice branch
        self.lattice_dense1 = nn.Dense(features=self.config['res_block_dim'])
        self.lattice_res_blocks = [ResidualBlock(features=self.config['res_block_dim']) 
                                 for _ in range(self.config['num_lattice_res_blocks'])]
        
        # Final layers after combining branches
        self.dense_final = nn.Dense(features=self.config['embedding_dim'] * 2)
        self.dense_output = nn.Dense(features=self.config['embedding_dim'])

    def __call__(self, x, space_group, lattice_vectors):
        space_group_onehot = jax.nn.one_hot(space_group, 230)
        lattice = jax.vmap(lambda lv: lv.ravel())(lattice_vectors)
        
        pos_input = jnp.concatenate([
            jnp.concatenate([jnp.real(x), jnp.imag(x)], axis=-1),  # Complex to real
            space_group_onehot[:, None, :].repeat(x.shape[1], axis=1)  # Add space group info
        ], axis=-1)
        
        lattice_input = jnp.concatenate([
            lattice[:, None, :].repeat(x.shape[1], axis=1),  # Add lattice info
            space_group_onehot[:, None, :].repeat(x.shape[1], axis=1)  # Add space group info
        ], axis=-1)

        # Position branch
        pos = self.pos_dense1(pos_input)
        pos = nn.relu(pos)
        pos = self.pos_dense2(pos)
        pos = nn.relu(pos)
        for i, res_block in enumerate(self.pos_res_blocks):
            pos = res_block(pos)
            
        # Lattice branch
        lat = self.lattice_dense1(lattice_input)
        lat = nn.relu(lat)
        for i, res_block in enumerate(self.lattice_res_blocks):
            lat = res_block(lat)
            
        x = pos * lat
        x = self.dense_final(x)
        x = nn.relu(x)
        x = self.dense_output(x)
        return x
    
class PretrainedPositionalEncoding(nn.Module):
    config: dict
    abc_combinations: jnp.ndarray
    adjacency_matrices: jnp.ndarray
    pretrained_state: Any

    def setup(self):
        self.mlp = MLP(self.config)
        self.params = self.param('mlp_params', lambda _: self.pretrained_state.params)

    @nn.compact
    def __call__(self, atom_positions, lattice_vectors, space_group):
        encodings = encode_positions(atom_positions, space_group, self.abc_combinations, self.adjacency_matrices)
        embeddings = self.mlp.apply(
            self.params,
            encodings, 
            space_group - 1, 
            lattice_vectors
        )
        # embeddings = self.mlp.apply(self.pretrained_state.params, encodings, space_group - 1, lattice_vectors)
        return embeddings

def encode_positions(atom_positions, space_group, abc_combinations, adjacency_matrices):
    encodings = jax.vmap(lambda pos: jnp.exp(1j * 2 * jnp.pi * jnp.dot(pos, abc_combinations.T)))(atom_positions)
    adjacency_matrix = adjacency_matrices[space_group - 1]
    return jax.vmap(jnp.matmul)(encodings, adjacency_matrix)

@jax.jit
def get_space_group_operations(space_group):
    return ALL_ROTATIONS[space_group - 1], ALL_TRANSLATIONS[space_group - 1]

@jax.jit
def precompute_orbits(positions, space_group):
    rotations, translations = get_space_group_operations(space_group)
    
    def compute_orbit(pos):
        new_positions = jax.vmap(lambda r, t: r @ pos + t)(rotations, translations)
        return new_positions
    return jax.vmap(compute_orbit)(positions)

@jax.jit
def min_distance_with_orbits(pos1_orbit, pos2_orbit, lattice, basis):
    def pbc_distance(p1, p2):
        diff = p1 - p2
        diff = diff - jnp.floor(diff + 0.5)
        cart_diff = diff @ lattice
        return jnp.linalg.norm(cart_diff, axis=1)
    pos1_orbit = pos1_orbit @ jnp.linalg.inv(basis.T)
    pos2_orbit = pos2_orbit @ jnp.linalg.inv(basis.T)
    distances = jax.vmap(lambda p2: pbc_distance(pos1_orbit[:1, :], p2))(pos2_orbit)
    return jnp.min(distances)

@jax.jit
def pairwise_distance_loss(embeddings, positions, lattice_vectors, space_groups):
    orbits = jax.vmap(lambda pos, sg: precompute_orbits(pos, sg))(positions, space_groups)
    def single_crystal_distances(orbit, lv, emb, basis):
        real_distance = min_distance_with_orbits(orbit[0], orbit[1], lv, basis)
        embedding_distance = jnp.sqrt(jnp.sum((emb[0] - emb[1]) ** 2) + 1e-6)
        squared_error = (real_distance - embedding_distance) ** 2
        absolute_error = jnp.abs(real_distance - embedding_distance)
        return squared_error, absolute_error
    crystal_errors = jax.vmap(single_crystal_distances)(orbits, lattice_vectors, embeddings, ALL_BASES[space_groups - 1])
    mse = jnp.mean(crystal_errors[0])
    mae = jnp.mean(crystal_errors[1])
    return mse, mae

@jax.jit
def train_step(state, batch, abc_combinations, adjacency_matrices, l2_weight):
    positions, lattice_vectors, space_groups = batch
    cartesian_positions = jax.vmap(lambda pos, basis: pos @ basis.T)(positions, ALL_BASES[space_groups - 1])
    
    def loss_fn(params):
        encodings = encode_positions(cartesian_positions, space_groups, abc_combinations, adjacency_matrices)
        embeddings = state.apply_fn(params, encodings, space_groups - 1, lattice_vectors)
        mse, mae = pairwise_distance_loss(embeddings, cartesian_positions, lattice_vectors, space_groups)
        
        # Add L2 regularization
        l2_loss = 0.0
        for param in jax.tree_util.tree_leaves(params):
            l2_loss += jnp.sum(param ** 2)
        
        total_loss = mse + l2_weight * l2_loss
        return total_loss, (mse, mae)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mse, mae)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, mse, mae

@jax.jit
def eval_step(state, batch, abc_combinations, adjacency_matrices):
    positions, lattice_vectors, space_groups = batch
    positions = jax.vmap(lambda pos, basis: pos @ basis.T)(positions, ALL_BASES[space_groups - 1])
    encodings = encode_positions(positions, space_groups, abc_combinations, adjacency_matrices)
    embeddings = state.apply_fn(state.params, encodings, space_groups - 1, lattice_vectors)
    mse, mae = pairwise_distance_loss(embeddings, positions, lattice_vectors, space_groups)
    return mse, mae

def load_data(data_dir, crystal_system="hexagonal", val_split=0.15, seed=42):
    rng = np.random.RandomState(seed)
    hex_space_groups = list(range(143, 195)) 
    cubic_space_groups = list(range(1, 143)) + list(range(195, 231))
    
    # Select space groups based on crystal system
    space_groups = hex_space_groups if crystal_system == "hexagonal" else cubic_space_groups
    
    # First pass: count total samples
    total_samples = 0
    for sg in space_groups:
        filename = os.path.join(data_dir, f"space_group_{sg}.npz")
        with np.load(filename) as data:
            total_samples += len(data['positions'])
    
    # Calculate train/val split indices
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    # Pre-allocate arrays
    train_positions = np.zeros((train_size, 2, 3), dtype=np.float32)
    train_lattice = np.zeros((train_size, 3, 3), dtype=np.float32)
    train_sg = np.zeros(train_size, dtype=np.int32)
    
    val_positions = np.zeros((val_size, 2, 3), dtype=np.float32)
    val_lattice = np.zeros((val_size, 3, 3), dtype=np.float32)
    val_sg = np.zeros(val_size, dtype=np.int32)
    
    # Second pass: fill arrays
    train_idx = 0
    val_idx = 0
    
    for sg in tqdm(space_groups):
        filename = os.path.join(data_dir, f"space_group_{sg}.npz")
        with np.load(filename) as data:
            positions = data['positions'].astype(np.float32)
            lattice_vectors = data['lattice_vectors'].astype(np.float32)
            space_groups = data['space_group'].astype(np.int32)
            
            # Generate random indices for this batch
            n_samples = len(positions)
            n_val = int(n_samples * val_split)
            indices = rng.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            # Add to training set
            curr_train_size = len(train_indices)
            train_positions[train_idx:train_idx + curr_train_size] = positions[train_indices]
            train_lattice[train_idx:train_idx + curr_train_size] = lattice_vectors[train_indices]
            train_sg[train_idx:train_idx + curr_train_size] = space_groups[train_indices]
            train_idx += curr_train_size
            
            # Add to validation set
            curr_val_size = len(val_indices)
            val_positions[val_idx:val_idx + curr_val_size] = positions[val_indices]
            val_lattice[val_idx:val_idx + curr_val_size] = lattice_vectors[val_indices]
            val_sg[val_idx:val_idx + curr_val_size] = space_groups[val_indices]
            val_idx += curr_val_size
    
    return (train_positions, train_lattice, train_sg), (val_positions, val_lattice, val_sg)

def create_train_state(config, model, input_shape, positions):
    key = jax.random.PRNGKey(seed=config['seed'])
    dummy_encodings = jnp.ones((input_shape[0], input_shape[1], input_shape[2]))
    dummy_space_groups = jnp.ones((input_shape[0]), dtype=jnp.int32)
    dummy_lattice_vectors = jnp.ones((input_shape[0], 3, 3))
    params = model.init(key, dummy_encodings, dummy_space_groups, dummy_lattice_vectors)

    total_steps = (len(positions) // config['batch_size']) * config['num_epochs']
    steps_per_epoch = len(positions) // config['batch_size']
    
    # Longer warmup period (5 epochs instead of 3)
    warmup_steps = steps_per_epoch * 5
    
    # Constant learning rate period (20% of total training)
    constant_steps = int(total_steps * 0.2)
    
    # Remaining steps for decay
    decay_steps = total_steps - warmup_steps - constant_steps

    warmup_fn = optax.linear_schedule(
        init_value=0.01 * config['learning_rate'],
        end_value=config['learning_rate'],
        transition_steps=warmup_steps
    )
    
    constant_fn = optax.constant_schedule(config['learning_rate'])
    
    cosine_decay_fn = optax.cosine_decay_schedule(
        init_value=config['learning_rate'],
        decay_steps=max(1, decay_steps),
        alpha=1e-4
    )
    
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, constant_fn, cosine_decay_fn],
        boundaries=[warmup_steps, warmup_steps + constant_steps]
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    
    return key, train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def train_model(train_data, val_data, abc_combinations, adjacency_matrices, config):
    config['fourier_dim'] = abc_combinations.shape[0]
    model = MLP(config)
    train_positions, train_lattice_vectors, train_space_groups = train_data
    val_positions, val_lattice_vectors, val_space_groups = val_data
    
    key, state = create_train_state(config, model, (1, 1, abc_combinations.shape[0]), train_positions)

    num_train_samples = len(train_positions)
    num_train_batches = num_train_samples // config['batch_size']
    
    num_val_samples = len(val_positions)
    num_val_batches = (num_val_samples + config['batch_size'] - 1) // config['batch_size']

    checkpoint_dir = f"/n/fs/cfs/crystal-fourier-transformer/mlp-ckpt/{config['ckpt']}"
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, num_train_samples)
        shuffled_positions = train_positions[permutation]
        shuffled_lattice_vectors = train_lattice_vectors[permutation]
        shuffled_space_groups = train_space_groups[permutation]

        train_mse = 0
        train_mae = 0
        for i in range(num_train_batches):
            batch_positions = shuffled_positions[i*config['batch_size']:(i+1)*config['batch_size']]
            batch_lattice_vectors = shuffled_lattice_vectors[i*config['batch_size']:(i+1)*config['batch_size']]
            batch_space_groups = shuffled_space_groups[i*config['batch_size']:(i+1)*config['batch_size']]
            
            batch = (batch_positions, batch_lattice_vectors, batch_space_groups)
            state, mse, mae = train_step(state, batch, abc_combinations, adjacency_matrices, config['l2_weight'])
            train_mse += mse
            train_mae += mae

        val_mse = 0
        val_mae = 0
        for i in range(num_val_batches):
            start_idx = i * config['batch_size']
            end_idx = min((i + 1) * config['batch_size'], num_val_samples)
            batch = (
                val_positions[start_idx:end_idx],
                val_lattice_vectors[start_idx:end_idx],
                val_space_groups[start_idx:end_idx]
            )
            mse, mae = eval_step(state, batch, abc_combinations, adjacency_matrices)
            val_mse += mse
            val_mae += mae
        
        avg_train_mse = train_mse / num_train_batches
        avg_train_mae = train_mae / num_train_batches
        avg_val_mse = val_mse / num_val_batches
        avg_val_mae = val_mae / num_val_batches
        
        print(f"Epoch {epoch+1}, Train MSE: {avg_train_mse:.4f}, Train MAE: {avg_train_mae:.4f}, "
              f"Val MSE: {avg_val_mse:.4f}, Val MAE: {avg_val_mae:.4f}")

        # Save best model checkpoint
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=state,
                step=epoch,
                prefix="best_model_",
                keep=1
            )
            
    return state

def load_trained_state(ckpt_dir, model):
    with open(os.path.join(ckpt_dir, "config.pkl"), "rb") as f:
        loaded_config = pickle.load(f)
    
    dummy_state = create_train_state(loaded_config, model, (1, 64, loaded_config['fourier_dim']), jnp.ones((1, 64, 3)))
    if isinstance(dummy_state, tuple):
        _, dummy_state = dummy_state

    restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=dummy_state, prefix="best_model_")
    return restored_state

def save_adjacency_matrices(adjacency_matrices, fourier_dim, crystal_system):
    """Save adjacency matrices to a compressed NPZ file."""
    save_path = f"data/adjacency_matrices_{crystal_system}_{fourier_dim}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, matrices=adjacency_matrices)
    print(f"Saved adjacency matrices to {save_path}")

def load_adjacency_matrices(fourier_dim, crystal_system):
    """Load adjacency matrices from NPZ file if it exists."""
    save_path = f"data/adjacency_matrices_{crystal_system}_{fourier_dim}.npz"
    if os.path.exists(save_path):
        print(f"Loading adjacency matrices from {save_path}")
        data = np.load(save_path)
        return jnp.array(data['matrices'])
    return None

def main():
    config = parse_args()
    fourier_dim = config['fourier_dim']
    crystal_system = config['crystal_system']
    
    # Use space group 1 for cubic, 168 for hexagonal
    reference_space_group = 1 if crystal_system == "cubic" else 168

    # Try to load precomputed adjacency matrices
    all_adjacency_matrices = load_adjacency_matrices(fourier_dim, crystal_system)
    abc_combinations = SpaceGraph(reference_space_group, fourier_dim).get_nodelist()
    print(f"Crystal system: {crystal_system}, reference space group: {reference_space_group}")
    print("abc_combinations shape: ", jnp.array(abc_combinations).shape)
    
    if all_adjacency_matrices is None:
        # Compute the correct points for each group
        all_adjacency_matrices = []

        print("Computing adjacency matrices for all space groups...")
        for sg in range(1, 231):
            sg_graph = SpaceGraph(sg, fourier_dim, points=abc_combinations)
            adj = sg_graph.get_adjacency_matrix().toarray().T
            all_adjacency_matrices.append(adj)

        all_adjacency_matrices = jnp.stack([adj for adj in all_adjacency_matrices])
        print("all_adjacency_matrices shape: ", all_adjacency_matrices.shape)
        save_adjacency_matrices(all_adjacency_matrices, fourier_dim, crystal_system)

    print("Loading data...")
    train_data, val_data = load_data(config['data_dir'], crystal_system=crystal_system)
    
    print("Starting training...")
    train_model(train_data, val_data, jnp.array(abc_combinations), all_adjacency_matrices, config)

if __name__ == "__main__":
    main()