# Crystal Fourier Transformer

A Transformer architecture for crystal property prediction using space group-invariant Fourier positional encodings that adapt to the input space group.

We provide pretrained models for four material properties:

| Property | Checkpoint Path | Description |
|----------|----------------|-------------|
| Bulk Modulus | `checkpoints/bulk_modulus` | Elastic bulk modulus (GPa) |
| Shear Modulus | `checkpoints/shear_modulus` | Elastic shear modulus (GPa) |
| Total Energy | `checkpoints/total_energy` | Total energy per atom (eV/atom) |
| Band Gap | `checkpoints/band_gap` | Electronic band gap (eV) |

## Directory Structure

```
crystal-fourier-transformer/
├── checkpoints/                    # Trained model checkpoints
│   ├── bulk_modulus/              # Bulk modulus prediction model
│   ├── shear_modulus/             # Shear modulus prediction model
│   ├── total_energy/              # Total energy prediction model
│   └── band_gap/                  # Band gap prediction model
├── pretrained_pos_enc/             # Pretrained positional encodings (128-dim)
│   ├── cubic_adjacency_matrices.npz
│   ├── hexagonal_adjacency_matrices.npz
│   ├── cubic_encoder/              # Pretrained MLP for cubic systems
│   └── hexagonal_encoder/          # Pretrained MLP for hexagonal systems
├── data/
│   └── get_materials.py           # Download data from Materials Project API
├── layers/
│   ├── attention.py               # Multi-head attention implementation
│   ├── feed_forward.py            # Feed-forward MLP layers
│   └── positional_encoding.py     # Positional encoding variants
├── pretrain/
│   ├── gen_data.py                # Generate synthetic crystal data
│   └── mlp.py                     # Pretrain positional encoding MLPs
├── utils/
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── gaussian_encoding.py       # Gaussian Fourier encoding computation
│   └── space_graphs.py            # Space group graph construction
├── model.py                       # CrystalFourierTransformer model
├── train.py                       # Training script
├── predict.py                     # Inference script
├── environment.yml                # Conda environment specification
└── README.md
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/PrincetonLIPS/crystal-fourier-transformer.git
cd crystal-fourier-transformer
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cft
```

## Quick Start: Using Pretrained Models

### Step 1: Prepare Your Data

Your data should be organized as follows:
```
my_crystals/
├── mp-123.cif          # CIF file for each material
├── mp-456.cif
├── ...
└── id_prop.csv         # CSV with material_id and property values
```

The `id_prop.csv` file should have the format:
```csv
mp-123,1.234
mp-456,5.678
...
```

You can download data from the Materials Project using the provided script:

```bash
export MP_API_KEY='YOUR_API_KEY'  # Get from https://materialsproject.org/
python -m data.get_materials --output_dir path/to/my_crystals
```

### Step 2: Preprocess the Data

Run the preprocessing to convert CIF files to the required format:

```python
from utils.data_processing import prepare_data

# This will process CIF files and cache the results
features, targets = prepare_data(
    root_dir='path/to/my_crystals',    # Directory with CIF files and id_prop.csv
    cache_dir='path/to/processed_data'  # Where to save processed data
)
```

### Step 3: Run Inference

Use the provided inference script:

```bash
# Predict using a pretrained model
python predict.py \
    --checkpoint checkpoints/bulk_modulus \
    --data_dir path/to/my_crystals \
    --output predictions.csv

# Or use pre-processed data
python predict.py \
    --checkpoint checkpoints/bulk_modulus \
    --cache_dir path/to/processed_data \
    --output predictions.csv
```

### Inference Script Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint directory |
| `--data_dir` | '' | Directory with CIF files and id_prop.csv |
| `--cache_dir` | '' | Directory with pre-processed pickle files |
| `--output` | predictions.csv | Output CSV file for predictions |
| `--batch_size` | 32 | Batch size for inference |
| `--pretrained_dir` | pretrained_pos_enc | Pretrained positional encodings directory (128-dim) |

## Training Your Own Model

### Step 1: Prepare Training Data

1. Obtain an API key from the [Materials Project](https://materialsproject.org/)

2. Set your API key as an environment variable:
```bash
export MP_API_KEY='YOUR_API_KEY'
```

3. Run the download script:
```bash
python -m data.get_materials
```

This downloads crystal structures to `data/materials/` by default, creating CIF files and an `id_prop.csv`.

### Step 2: Train the Model

Basic training with default paths (uses `data/materials/` from Step 1):
```bash
python train.py --ckpt_dir checkpoints/my_model
```

Or specify custom directories:
```bash
python train.py \
    --data_dir data/materials \
    --cache_dir data/materials-cache \
    --ckpt_dir checkpoints/my_model \
    --pretrained_pos_enc pretrained_pos_enc
```

**Note:** If `--data_dir` doesn't exist or contains no CIF files, training will fail. The `--cache_dir` is created automatically to store preprocessed data for faster subsequent runs.

## Pretraining Positional Encodings

If you want to train positional encodings from scratch instead of using the provided pretrained encodings:

```bash
# Generate synthetic crystal data
python -m pretrain.gen_data

# Train cubic system encoder (space groups 1-142, 195-230)
python -m pretrain.mlp --data_dir data/synthetic_crystals_cubic --ckpt my_cubic_encoder

# Train hexagonal system encoder (space groups 143-194)
python -m pretrain.mlp --data_dir data/synthetic_crystals_hex --ckpt my_hexagonal_encoder
```

## Model Architecture

The Crystal Fourier Transformer consists of:

1. **Atom Embedding**: Learnable embeddings for atomic numbers (1-100)

2. **Positional Encoding**: Space group-aware Fourier basis functions
   - Separate encoders for cubic and hexagonal crystal systems
   - Pretrained MLP maps Fourier coefficients to embedding dimension (128-dim)
   - Optional Gaussian density convolution for smoothness

3. **Transformer Blocks**: Pre-LayerNorm architecture with:
   - Multi-head self-attention
   - SiLU-activated feed-forward network
   - Residual connections

4. **Output Head**: Mean pooling + MLP for property prediction

## Data Format

The training pipeline expects:
- A directory containing `.cif` files for each material (named as `{material_id}.cif`)
- An `id_prop.csv` file with columns: `material_id`, `property_value` (no header)

The preprocessing pipeline (`utils/data_processing.py`) handles:
- Parsing CIF files using pymatgen
- Converting structures to fractional coordinates
- Padding to a maximum of 444 atoms per structure
- Caching preprocessed data as pickle files

## Citation

Feel free to open an issue for any questions or suggestions. If you find this code useful, please consider citing: 

```bibtex
@article{zhang2025sginvariance,
  title={A Single Architecture for Representing Invariance Under Any Space Group},
  author={Zhang, Cindy Y and Ertekin, Elif and Orbanz, Peter and Adams, Ryan P},
  journal={arXiv preprint arXiv:2512.13989},
  year={2025}
}
```
