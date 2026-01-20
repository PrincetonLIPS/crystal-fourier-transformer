import os
import csv
import jax.numpy as jnp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm

def load_crystal_data(root_dir: str, cache_file: str = None) -> List[Tuple[str, float, int, Structure]]:
    """
    Load crystal data from the root directory, with caching.
    
    Args:
        root_dir (str): Path to the root directory containing id_prop.csv and CIF files.
        cache_file (str): Path to the cache file for saving/loading processed data.
    
    Returns:
        List[Tuple[str, float, int, Structure]]: List of tuples containing CIF ID, target property, space group number, and pymatgen Structure.
    """
    if cache_file is None:
        cache_file = os.path.join(root_dir, 'crystal_data.pkl')
    
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading crystal data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            crystal_data = pickle.load(f)
        print(f"Loaded {len(crystal_data)} crystals from cache")
        return crystal_data

    # Load data from CSV and CIF files
    id_prop_file = os.path.join(root_dir, 'id_prop.csv')
    
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]
    
    crystal_data = []
    for cif_id, target in id_prop_data:
        cif_path = os.path.join(root_dir, f'{cif_id}.cif')
        structure = Structure.from_file(cif_path)
        space_group = SpacegroupAnalyzer(structure).get_space_group_number()
        target_value = float(target)
        crystal_data.append((cif_id, target_value, space_group, structure))
    
    # Cache the processed data
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(crystal_data, f)
    print(f"Cached {len(crystal_data)} crystals to {cache_file}")
    
    return crystal_data


def pre_process_data(crystal_data: List[Tuple[str, float, int, Structure]]) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, List[str]]:
    """
    Preprocess the crystal data for use in the Crystal Fourier Transformer.
    
    Args:
        crystal_data (List[Tuple[str, float, int, Structure]]): List of crystal data tuples.
    
    Returns:
        Tuple[Dict[str, jnp.ndarray], jnp.ndarray, List[str]]: 
            - Dictionary containing 'atom_numbers', 'positions', 'masks', etc.
            - Array of target properties
    """
    max_atoms = 444
    
    data = {
        'atom_numbers': [],
        'positions': [],
        'lattice_matrices': [],
        'space_groups': [],
        'masks': [],
    }
    targets = []

    for cif_id, target, space_group, structure in tqdm(crystal_data, desc="Processing crystals"):
        num_atoms = len(structure)
        
        atom_nums = jnp.array([site.specie.number for site in structure], dtype=jnp.int32)
        frac_coords = jnp.array([site.frac_coords for site in structure])
        lattice = structure.lattice._matrix
        
        # Create mask: 1 for actual atoms, 0 for padding
        mask = jnp.concatenate([jnp.ones(num_atoms), jnp.zeros(max_atoms - num_atoms)])
        
        data['atom_numbers'].append(jnp.pad(atom_nums, (0, max_atoms - num_atoms)))
        data['positions'].append(jnp.pad(frac_coords, ((0, max_atoms - num_atoms), (0, 0))))
        data['lattice_matrices'].append(jnp.array(lattice))
        data['space_groups'].append(space_group)
        data['masks'].append(mask)
        
        targets.append(target)

    for key in data:
        data[key] = jnp.array(data[key])

    return data, jnp.array(targets)


def prepare_data_in_batches(root_dir: str, cache_dir: str, batch_size: int = 20000):
    """
    Prepare and cache the pre-processed crystal data in batches.
    
    Args:
        root_dir (str): Path to the root directory containing id_prop.csv and CIF files.
        cache_dir (str): Path to the directory where batched data will be saved.
        batch_size (int): Number of crystals to process in each batch.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading crystal data from {root_dir}")
    crystal_data = load_crystal_data(root_dir)
    
    total_crystals = len(crystal_data)
    num_batches = (total_crystals + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_crystals)
        
        print(f"Processing batch {i+1}/{num_batches}")
        batch_data = crystal_data[start_idx:end_idx]
        
        preprocessed_batch = pre_process_data(batch_data)
        
        batch_file = os.path.join(cache_dir, f'batch_{i}.pkl')
        with open(batch_file, 'wb') as f:
            pickle.dump(preprocessed_batch, f)
        
        print(f"Saved batch {i+1} to {batch_file}")


def prepare_data(root_dir: str, cache_dir: str = 'data/materials-cache') -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Prepare and cache the pre-processed crystal data, or load from existing batches.
    
    Args:
        root_dir (str): Path to the root directory containing id_prop.csv and CIF files.
        cache_dir (str): Path to the directory for caching preprocessed batched data.
    
    Returns:
        Tuple[Dict[str, jnp.ndarray], jnp.ndarray]: Combined pre-processed data from all batches.
    """
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        print(f"Processing data from {root_dir} into batches at {cache_dir}")
        prepare_data_in_batches(root_dir, cache_dir)
    
    print("Loading pre-processed data from batches")
    combined_data = {
        'atom_numbers': [],
        'positions': [],
        'lattice_matrices': [],
        'space_groups': [],
        'masks': [],
    }
    combined_targets = []
    
    for batch_file in sorted(os.listdir(cache_dir)):
        if batch_file.endswith('.pkl'):
            with open(os.path.join(cache_dir, batch_file), 'rb') as f:
                loaded_data = pickle.load(f)

            if isinstance(loaded_data, dict) and 'features' in loaded_data:
                # Handle data from preprocess_mask.py
                batch_data = loaded_data['features']
                batch_targets = loaded_data['targets']
            else:
                # Handle original batch format
                batch_data, batch_targets = loaded_data
            
            for key in combined_data:
                combined_data[key].append(batch_data[key])
            
            combined_targets.append(batch_targets)
    
    for key in combined_data:
        combined_data[key] = jnp.concatenate(combined_data[key], axis=0)
    
    combined_targets = jnp.concatenate(combined_targets, axis=0)
    print("Total number of crystals: ", len(combined_targets))
    
    return combined_data, combined_targets