"""
This script loads a trained CFT model and makes predictions on crystal structures.
It supports both pre-processed data and raw CIF files.

Usage:
    # Predict on a directory of CIF files
    python predict.py --checkpoint checkpoints/bulk_modulus --data_dir data/my_crystals --output predictions.csv
    
    # Predict on pre-processed data
    python predict.py --checkpoint checkpoints/bulk_modulus --cache_dir data/my_processed --output predictions.csv
"""

import jax
import jax.numpy as jnp
import json
import pickle
import argparse
import numpy as np
import os
import csv
import optax
from model import CrystalFourierTransformer
from utils.data_processing import prepare_data, load_crystal_data, pre_process_data
from utils.space_graphs import SpaceGraph
from pretrain.mlp import MLP
from flax.training import train_state, checkpoints
from utils.gaussian_encoding import compute_batch_encodings


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the Crystal Fourier Transformer")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint directory (e.g., checkpoints/bulk_modulus)")
    parser.add_argument('--data_dir', type=str, default='',
                        help="Directory containing CIF files and id_prop.csv")
    parser.add_argument('--cache_dir', type=str, default='',
                        help="Directory containing pre-processed pickle files")
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help="Output CSV file for predictions")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument('--pretrained_dir', type=str, default='pretrained_pos_enc',
                        help="Directory containing pretrained positional encodings (128-dim)")
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.cache_dir:
        parser.error("At least one of --data_dir or --cache_dir must be specified")
    
    return args


def load_pretrained_state(checkpoint_dir):
    """Load pretrained positional encoding MLP state."""
    with open(f"{checkpoint_dir}/config.pkl", "rb") as f:
        config = pickle.load(f)
    mlp = MLP(config)
    
    from pretrain.mlp import create_train_state
    dummy_state = create_train_state(config, mlp, (1, 64, config['fourier_dim']), jnp.ones((1, 64, 3)))
    if isinstance(dummy_state, tuple):
        _, dummy_state = dummy_state
    
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=dummy_state, prefix="best_model_")
    return config, restored_state


def get_encoding_components(pretrained_dir):
    """Load positional encoding components from pretrained directory."""
    # Convert to absolute path if relative
    pretrained_dir = os.path.abspath(pretrained_dir)
    
    cubic_adj_path = os.path.join(pretrained_dir, 'cubic_adjacency_matrices.npz')
    hexagonal_adj_path = os.path.join(pretrained_dir, 'hexagonal_adjacency_matrices.npz')
    
    cubic_adj_matrices = jnp.array(np.load(cubic_adj_path)['matrices'])
    hexagonal_adj_matrices = jnp.array(np.load(hexagonal_adj_path)['matrices'])
    
    cubic_abc_combinations = jnp.array(SpaceGraph(1, 200).get_nodelist())
    hexagonal_abc_combinations = jnp.array(SpaceGraph(168, 300).get_nodelist())
    
    cubic_encoder_dir = os.path.join(pretrained_dir, 'cubic_encoder')
    hexagonal_encoder_dir = os.path.join(pretrained_dir, 'hexagonal_encoder')
    
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


def load_model(checkpoint_dir, pretrained_dir):
    """Load a trained Crystal Fourier Transformer model."""
    # Convert to absolute paths if relative
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    (
        cubic_adj_matrices,
        hexagonal_adj_matrices,
        cubic_abc_combinations,
        hexagonal_abc_combinations,
        cubic_encoding_config,
        hexagonal_encoding_config,
        cubic_pretrained_state,
        hexagonal_pretrained_state,
    ) = get_encoding_components(pretrained_dir)
    
    model = CrystalFourierTransformer(
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
    
    class TrainStateWithBatchStats(train_state.TrainState):
        batch_stats: dict
    
    # Initialize model with dummy input
    key = jax.random.PRNGKey(0)
    dummy_atom_numbers = jnp.ones((1, 444), dtype=jnp.int32)
    dummy_positions = jnp.zeros((1, 444, 3))
    dummy_lattice = jnp.eye(3)[None, :, :]
    dummy_space_groups = jnp.array([1])
    dummy_masks = jnp.ones((1, 444))
    
    init_args = {
        'atom_numbers': dummy_atom_numbers,
        'positions': dummy_positions,
        'lattice_matrices': dummy_lattice,
        'space_groups': dummy_space_groups,
        'masks': dummy_masks,
        'training': False
    }
    
    if config.get('gaussian_encoding', False):
        gaussian_dim = config.get('gaussian_actual_dim', 128)
        init_args['precomputed_positional_encodings'] = jnp.zeros((1, 444, gaussian_dim))
    
    variables = model.init({'params': key}, **init_args)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    tx = optax.adam(learning_rate=1e-4)
    
    dummy_state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )
    
    state = checkpoints.restore_checkpoint(checkpoint_dir, dummy_state)
    
    return model, state, config


def predict_batch(model, state, batch, config):
    """Run inference on a batch of data."""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    
    model_inputs = {
        'atom_numbers': batch['atom_numbers'],
        'positions': batch['positions'],
        'lattice_matrices': batch['lattice_matrices'],
        'space_groups': batch['space_groups'],
        'masks': batch['masks'],
        'training': False
    }
    
    if config.get('gaussian_encoding', False):
        model_inputs['precomputed_positional_encodings'] = batch['positional_encodings']
    
    predictions = model.apply(variables, **model_inputs, mutable=False)
    return jnp.squeeze(predictions)


def main():
    args = parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, state, config = load_model(args.checkpoint, args.pretrained_dir)
    print("Model loaded successfully.")
    
    # Load data
    if args.cache_dir and os.path.exists(args.cache_dir) and len(os.listdir(args.cache_dir)) > 0:
        print(f"Loading pre-processed data from {args.cache_dir}...")
        features, targets = prepare_data(root_dir=args.data_dir if args.data_dir else '', cache_dir=args.cache_dir)
        material_ids = None
    else:
        print(f"Processing CIF files from {args.data_dir}...")
        crystal_data = load_crystal_data(args.data_dir, cache_file=os.path.join(args.data_dir, 'crystal_data.pkl'))
        features, targets = pre_process_data(crystal_data)
        material_ids = [cd[0] for cd in crystal_data]
    
    num_samples = len(targets)
    print(f"Loaded {num_samples} samples.")
    
    # Compute Gaussian encodings if needed
    if config.get('gaussian_encoding', False):
        gaussian_dim = config.get('gaussian_actual_dim', 128)
        gaussian_sigma = config.get('gaussian_sigma', 2.0)
        
        print(f"Computing Gaussian Fourier positional encodings (dim={gaussian_dim})...")
        positional_encodings, _, _ = compute_batch_encodings(
            features['positions'],
            features['lattice_matrices'],
            features['space_groups'],
            features['masks'],
            gaussian_dim,
            gaussian_sigma,
            k_vectors_frac=None,
        )
        features['positional_encodings'] = positional_encodings
        print("Gaussian encodings computed.")
    
    # Run inference
    print("Running inference...")
    all_predictions = []
    
    for i in range(0, num_samples, args.batch_size):
        end_idx = min(i + args.batch_size, num_samples)
        batch = {k: v[i:end_idx] for k, v in features.items()}
        batch['targets'] = targets[i:end_idx]
        
        predictions = predict_batch(model, state, batch, config)
        all_predictions.append(np.asarray(predictions))
        
        if (i + args.batch_size) % (args.batch_size * 10) == 0:
            print(f"  Processed {min(i + args.batch_size, num_samples)}/{num_samples} samples...")
    
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Compute metrics
    targets_np = np.asarray(targets)
    mae = np.mean(np.abs(predictions - targets_np))
    mse = np.mean((predictions - targets_np) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nResults:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE:  {mse:.4f}")
    
    # Save predictions
    print(f"\nSaving predictions to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        if material_ids:
            writer.writerow(['material_id', 'target', 'prediction', 'error'])
            for mid, target, pred in zip(material_ids, targets_np, predictions):
                writer.writerow([mid, target, pred, pred - target])
        else:
            writer.writerow(['index', 'target', 'prediction', 'error'])
            for i, (target, pred) in enumerate(zip(targets_np, predictions)):
                writer.writerow([i, target, pred, pred - target])
    
    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()
