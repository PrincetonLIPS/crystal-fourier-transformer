import jax
import jax.numpy as jnp
import numpy as np
import json
import optax
import time
import pickle
import argparse
import csv
import os
from functools import partial
from model import CrystalFourierTransformer
from utils.data_processing import prepare_data
from utils.gaussian_encoding import compute_batch_encodings
from utils.space_graphs import SpaceGraph
from pretrain.mlp import load_trained_state, MLP
from jax import random
from flax.training import train_state, checkpoints

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Crystal Fourier Transformer")
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of embeddings")
    parser.add_argument('--num_attn_blocks', type=int, default=4, help="Number of attention blocks")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--ff_dim', type=int, default=512, help="Feed-forward hidden dimension")
    parser.add_argument('--final_hidden_1', type=int, default=2048, help="Final MLP hidden dimension 1")
    parser.add_argument('--final_hidden_2', type=int, default=256, help="Final MLP hidden dimension 2")
    parser.add_argument('--gaussian_encoding', action='store_true', help="Use Gaussian Fourier positional encodings")
    parser.add_argument('--gaussian_sigma', type=float, default=2, help="Sigma for Gaussian density (Angstroms)")

    # Training parameters
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/test')
    parser.add_argument('--data_dir', type=str, default='data/materials')
    parser.add_argument('--cache_dir', type=str, default='data/materials-cache')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_fraction', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of data for testing')
    parser.add_argument('--pretrained_pos_enc', type=str, default='pretrained_pos_enc', help='Pretrained positional encodings directory (128-dim)')
    
    args = parser.parse_args()
    return vars(args)

def create_learning_rate_fn(config, num_train_examples):
    """Create a cosine decay learning rate schedule with linear warmup."""
    steps_per_epoch = num_train_examples // config['batch_size']
    total_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = config['warmup_epochs'] * steps_per_epoch

    warmup_fn = optax.linear_schedule(
        init_value=0.1 * config['learning_rate'],
        end_value=config['learning_rate'],
        transition_steps=warmup_steps
    )
    cosine_decay_fn = optax.cosine_decay_schedule(
        init_value=config['learning_rate'],
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_decay_fn],
        boundaries=[warmup_steps]
    )

    return schedule_fn

def save_model_config(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f)

@partial(jax.jit, static_argnums=(0, 4))
def train_step(apply_fn, state, batch, dropout_rng, use_gaussian_encoding):
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        
        model_inputs = {
            'atom_numbers': batch['atom_numbers'],
            'positions': batch['positions'],
            'lattice_matrices': batch['lattice_matrices'],
            'space_groups': batch['space_groups'],
            'masks': batch['masks'],
            'training': True,
        }
        if use_gaussian_encoding:
            model_inputs['precomputed_positional_encodings'] = batch['positional_encodings']

        predictions, new_model_state = apply_fn(
            variables,
            **model_inputs,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        predictions = jnp.squeeze(predictions)
        mse_loss = jnp.mean((predictions - batch['targets']) ** 2)
        
        return mse_loss, (predictions, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (predictions, new_model_state)), grads = grad_fn(state.params)
    
    updated_batch_stats = new_model_state.get('batch_stats', state.batch_stats)
    state = state.apply_gradients(
        grads=grads,
        batch_stats=updated_batch_stats
    )
    
    metrics = {'loss': loss, 'mae': jnp.mean(jnp.abs(predictions - batch['targets']))}
    return state, metrics

@partial(jax.jit, static_argnums=(0, 4))
def eval_step(apply_fn, params, batch_stats, batch, use_gaussian_encoding):
    variables = {'params': params, 'batch_stats': batch_stats}

    model_inputs = {
        'atom_numbers': batch['atom_numbers'],
        'positions': batch['positions'],
        'lattice_matrices': batch['lattice_matrices'],
        'space_groups': batch['space_groups'],
        'masks': batch['masks'],
        'training': False
    }
    if use_gaussian_encoding:
        model_inputs['precomputed_positional_encodings'] = batch['positional_encodings']
        
    predictions = apply_fn(variables, **model_inputs, mutable=False)
    predictions = jnp.squeeze(predictions)
    loss = jnp.mean((predictions - batch['targets']) ** 2)
    mae = jnp.mean(jnp.abs(predictions - batch['targets']))
    return loss, mae

def create_train_state(config, cft, train_features, key):
    key, init_key = random.split(key)
    
    init_args = {
        'atom_numbers': train_features['atom_numbers'][:1],
        'positions': train_features['positions'][:1],
        'lattice_matrices': train_features['lattice_matrices'][:1],
        'space_groups': train_features['space_groups'][:1],
        'masks': train_features['masks'][:1],
        'training': False
    }

    # If using Gaussian encodings, pass a dummy positional tensor
    if config['gaussian_encoding']:
        gaussian_dim = config.get('gaussian_actual_dim', 128)
        dummy_enc = jnp.zeros((1, init_args['positions'].shape[1], gaussian_dim), dtype=jnp.float32)
        init_args['precomputed_positional_encodings'] = dummy_enc

    variables = cft.init({'params': init_key}, **init_args)
    
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    learning_rate_fn = create_learning_rate_fn(config, len(train_features['atom_numbers']))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adamw(learning_rate=learning_rate_fn, weight_decay=config['weight_decay'])
    )
    
    class TrainStateWithBatchStats(train_state.TrainState):
        batch_stats: dict

    return TrainStateWithBatchStats.create(
        apply_fn=cft.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats
    )

def load_pretrained_state(checkpoint_dir):
    with open(f"{checkpoint_dir}/config.pkl", "rb") as f:
        config = pickle.load(f)
    mlp = MLP(config)
    state = load_trained_state(checkpoint_dir, mlp)
    return config, state

def get_encoding_components(pretrained_pos_enc='pretrained_pos_enc'):
    """Load positional encoding components from pretrained directory.

    Args:
        pretrained_pos_enc: Path to directory containing pretrained positional encodings

    Returns:
        Tuple of encoding components for model initialization
    """
    # Convert to absolute path if relative
    pretrained_pos_enc = os.path.abspath(pretrained_pos_enc)
    
    # Load adjacency matrices
    cubic_adj_path = os.path.join(pretrained_pos_enc, 'cubic_adjacency_matrices.npz')
    hexagonal_adj_path = os.path.join(pretrained_pos_enc, 'hexagonal_adjacency_matrices.npz')
    cubic_adj_matrices = jnp.array(np.load(cubic_adj_path)['matrices'])
    hexagonal_adj_matrices = jnp.array(np.load(hexagonal_adj_path)['matrices'])
    
    # Get Fourier basis combinations
    cubic_abc_combinations = jnp.array(SpaceGraph(1, 200).get_nodelist())
    hexagonal_abc_combinations = jnp.array(SpaceGraph(168, 300).get_nodelist())
    
    # Load pretrained encoder states
    cubic_encoder_dir = os.path.join(pretrained_pos_enc, 'cubic_encoder')
    hexagonal_encoder_dir = os.path.join(pretrained_pos_enc, 'hexagonal_encoder')
    cubic_encoding_config, cubic_pretrained_state = load_pretrained_state(cubic_encoder_dir)
    hexagonal_encoding_config, hexagonal_pretrained_state = load_pretrained_state(hexagonal_encoder_dir)

    return (
        cubic_adj_matrices,
        hexagonal_adj_matrices,
        cubic_abc_combinations,
        hexagonal_abc_combinations,
        cubic_encoding_config,
        hexagonal_encoding_config,
        cubic_pretrained_state,
        hexagonal_pretrained_state,
    )

def main():
    config = parse_args()
    
    # Convert paths to absolute
    config['ckpt_dir'] = os.path.abspath(config['ckpt_dir'])
    config['pretrained_pos_enc'] = os.path.abspath(config['pretrained_pos_enc'])
    
    (
        cubic_adj_matrices,
        hexagonal_adj_matrices,
        cubic_abc_combinations,
        hexagonal_abc_combinations,
        cubic_encoding_config,
        hexagonal_encoding_config,
        cubic_pretrained_state,
        hexagonal_pretrained_state,
    ) = get_encoding_components(config['pretrained_pos_enc'])

    key = random.PRNGKey(config['seed'])
    features, targets = prepare_data(root_dir=config['data_dir'], cache_dir=config['cache_dir'])

    # Build train/val/test split
    key, shuffle_key = random.split(key)
    all_idx = jnp.arange(len(targets))
    shuffled_indices = random.permutation(shuffle_key, all_idx)
    n_total = int(shuffled_indices.shape[0])
    test_size = int(float(config['test_fraction']) * n_total)
    val_size = int(float(config['val_fraction']) * n_total)
    train_size = n_total - val_size - test_size

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    print(f"Total samples: {len(targets)} | Train: {train_indices.shape[0]} | Val: {val_indices.shape[0]} | Test: {test_indices.shape[0]}")

    train_features = {k: v[train_indices] for k, v in features.items()}
    train_targets = targets[train_indices]
    val_features = {k: v[val_indices] for k, v in features.items()}
    val_targets = targets[val_indices]
    test_features = {k: v[test_indices] for k, v in features.items()}
    test_targets = targets[test_indices]

    # Compute Gaussian Fourier positional encodings if enabled
    if config['gaussian_encoding']:
        # Compute target dimension: embedding_dim - 128 (pretrained encoding dim)
        gaussian_dim = config['embedding_dim'] - 128
        if gaussian_dim <= 0:
            gaussian_dim = 128
        
        print(f"Computing Gaussian Fourier positional encodings (dim={gaussian_dim})...")
        train_enc, k_vectors_frac, gaussian_actual_dim = compute_batch_encodings(
            train_features['positions'],
            train_features['lattice_matrices'],
            train_features['space_groups'],
            train_features['masks'],
            gaussian_dim,
            config['gaussian_sigma'],
            k_vectors_frac=None,
        )
        val_enc, _, _ = compute_batch_encodings(
            val_features['positions'],
            val_features['lattice_matrices'],
            val_features['space_groups'],
            val_features['masks'],
            gaussian_actual_dim,
            config['gaussian_sigma'],
            k_vectors_frac,
        )
        test_enc, _, _ = compute_batch_encodings(
            test_features['positions'],
            test_features['lattice_matrices'],
            test_features['space_groups'],
            test_features['masks'],
            gaussian_actual_dim,
            config['gaussian_sigma'],
            k_vectors_frac,
        )
        train_features['positional_encodings'] = train_enc
        val_features['positional_encodings'] = val_enc
        test_features['positional_encodings'] = test_enc
        config['gaussian_actual_dim'] = gaussian_actual_dim
        print(f"Done (actual dim: {gaussian_actual_dim}).")

    cft = CrystalFourierTransformer(
        config, 
        jnp.array(cubic_abc_combinations), 
        jnp.array(hexagonal_abc_combinations), 
        cubic_adj_matrices, 
        hexagonal_adj_matrices, 
        cubic_pretrained_state, 
        hexagonal_pretrained_state, 
        cubic_encoding_config, 
        hexagonal_encoding_config,
    )
    state = create_train_state(config, cft, train_features, key)
    save_model_config(config, config['ckpt_dir'])

    # Training loop
    best_val_mae = float('inf')
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        train_losses, train_maes = [], []
        key, shuffle_key = random.split(key)
        train_indices = random.permutation(shuffle_key, train_indices)
        
        for i in range(0, len(train_indices), config['batch_size']):
            batch_indices = train_indices[i:i+config['batch_size']]
            batch = {k: v[batch_indices] for k, v in train_features.items()}
            batch['targets'] = train_targets[batch_indices]
            
            key, dropout_key = random.split(key)
            state, metrics = train_step(cft.apply, state, batch, dropout_key, config['gaussian_encoding'])
            
            train_losses.append(metrics['loss'])
            train_maes.append(metrics['mae'])

        val_losses, val_maes = [], []
        for i in range(0, len(val_targets), config['batch_size']):
            batch = {k: v[i:i+config['batch_size']] for k, v in val_features.items()}
            batch['targets'] = val_targets[i:i+config['batch_size']]
            loss, mae = eval_step(cft.apply, state.params, state.batch_stats, batch, config['gaussian_encoding'])
            val_losses.append(loss)
            val_maes.append(mae)

        train_loss, train_mae = jnp.mean(jnp.array(train_losses)), jnp.mean(jnp.array(train_maes))
        val_loss, val_mae = jnp.mean(jnp.array(val_losses)), jnp.mean(jnp.array(val_maes))
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, "
              f"Time: {epoch_time:.2f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoints.save_checkpoint(config['ckpt_dir'], state, step=epoch, keep=2)

    # Final evaluation on test set using best checkpoint
    state = checkpoints.restore_checkpoint(config['ckpt_dir'], state)
    test_losses, test_maes = [], []
    all_preds, all_tgts, all_sgs = [], [], []
    
    for i in range(0, len(test_targets), config['batch_size']):
        batch = {k: v[i:i+config['batch_size']] for k, v in test_features.items()}
        batch['targets'] = test_targets[i:i+config['batch_size']]
        
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        model_inputs = {
            'atom_numbers': batch['atom_numbers'],
            'positions': batch['positions'],
            'lattice_matrices': batch['lattice_matrices'],
            'space_groups': batch['space_groups'],
            'masks': batch['masks'],
            'training': False
        }
        if config['gaussian_encoding']:
            model_inputs['precomputed_positional_encodings'] = batch['positional_encodings']
        predictions = cft.apply(variables, **model_inputs, mutable=False)
        predictions = jnp.squeeze(predictions)

        loss = jnp.mean((predictions - batch['targets']) ** 2)
        mae = jnp.mean(jnp.abs(predictions - batch['targets']))
        test_losses.append(loss)
        test_maes.append(mae)

        all_preds.append(np.asarray(predictions))
        all_tgts.append(np.asarray(batch['targets']))
        all_sgs.append(np.asarray(batch['space_groups']))

    test_loss, test_mae = jnp.mean(jnp.array(test_losses)), jnp.mean(jnp.array(test_maes))
    print(f"Final Test Loss: {test_loss:.4f}, Final Test MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()
