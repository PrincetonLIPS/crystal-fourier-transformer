import flax.linen as nn
import jax.numpy as jnp
from layers.attention import AttentionBlock
from layers.feed_forward import FeedForward
from pretrain.mlp import PretrainedPositionalEncoding
from typing import Any

class CrystalFourierTransformer(nn.Module):
    """Crystal Fourier Transformer for crystal property prediction.
    
    Uses space group-aware Fourier positional encodings to represent atomic positions
    in a way that respects crystallographic symmetries.
    
    Args:
        config: Model configuration dictionary containing embedding_dim, num_attn_blocks,
                num_heads, ff_dim, final_hidden_1, final_hidden_2,
                gaussian_encoding (bool), dropout_rate.
        cubic_abc_combinations: Fourier basis vectors for cubic space groups.
        hexagonal_abc_combinations: Fourier basis vectors for hexagonal space groups.
        cubic_adj_matrices: Adjacency matrices encoding space group symmetries (cubic).
        hexagonal_adj_matrices: Adjacency matrices encoding space group symmetries (hexagonal).
        cubic_pretrained_state: Pretrained MLP state for cubic positional encodings.
        hexagonal_pretrained_state: Pretrained MLP state for hexagonal positional encodings.
        cubic_encoding_config: Config for cubic positional encoding MLP.
        hexagonal_encoding_config: Config for hexagonal positional encoding MLP.
    """
    config: dict
    cubic_abc_combinations: jnp.ndarray
    hexagonal_abc_combinations: jnp.ndarray
    cubic_adj_matrices: jnp.ndarray
    hexagonal_adj_matrices: jnp.ndarray
    cubic_pretrained_state: Any = None
    hexagonal_pretrained_state: Any = None
    cubic_encoding_config: dict = None
    hexagonal_encoding_config: dict = None

    def setup(self):
        self.atom_embedding = nn.Embed(
            num_embeddings=101,  # Elements 1-100 plus padding
            features=self.config['embedding_dim']
        )
        
        # Fourier-based positional encodings with pretrained weights
        self.cubic_positional_encoding = PretrainedPositionalEncoding(
            config=self.cubic_encoding_config,
            abc_combinations=self.cubic_abc_combinations,
            adjacency_matrices=self.cubic_adj_matrices,
            pretrained_state=self.cubic_pretrained_state
        )
        self.hexagonal_positional_encoding = PretrainedPositionalEncoding(
            config=self.hexagonal_encoding_config,
            abc_combinations=self.hexagonal_abc_combinations,
            adjacency_matrices=self.hexagonal_adj_matrices,
            pretrained_state=self.hexagonal_pretrained_state
        )
        
        # Projection layer for concatenated positional encodings (Gaussian + learned)
        if self.config.get('gaussian_encoding', False):
            self.pos_encoding_proj = nn.Dense(self.config['embedding_dim'])
        
        self.input_norm = nn.LayerNorm()
        self.attention_blocks = [
            AttentionBlock(self.config) 
            for _ in range(self.config['num_attn_blocks'])
        ]
        self.final_ff = FeedForward(self.config)

    def __call__(self, atom_numbers, positions, lattice_matrices, space_groups, masks, 
                 training=True, rngs=None, precomputed_positional_encodings=None):
        """Forward pass of the Crystal Fourier Transformer.
        
        Args:
            atom_numbers: (B, N) atomic numbers for each atom.
            positions: (B, N, 3) fractional coordinates.
            lattice_matrices: (B, 3, 3) lattice vectors.
            space_groups: (B,) space group numbers (1-230).
            masks: (B, N) mask for valid atoms (1) vs padding (0).
            training: Whether in training mode (enables dropout).
            rngs: Random number generators for dropout.
            precomputed_positional_encodings: Optional (B, N, D) precomputed Gaussian encodings.
            
        Returns:
            (B, 1) predicted property values.
        """
        atom_embeddings = self.atom_embedding(atom_numbers)

        # Determine which samples use hexagonal vs cubic encodings
        is_hexagonal = (space_groups >= 143) & (space_groups <= 194)

        # Compute learned positional encodings
        cubic_pos_encodings = self.cubic_positional_encoding(
            positions, lattice_matrices, space_groups
        )
        hexagonal_pos_encodings = self.hexagonal_positional_encoding(
            positions, lattice_matrices, space_groups
        )
        learned_pos_encodings = jnp.where(
            is_hexagonal[:, None, None],
            hexagonal_pos_encodings,
            cubic_pos_encodings,
        )

        if self.config.get('gaussian_encoding', False) and precomputed_positional_encodings is not None:
            # Gaussian mode: concatenate precomputed Gaussian encodings with learned encodings
            pos_encodings = jnp.concatenate(
                [precomputed_positional_encodings, learned_pos_encodings], axis=-1
            )
            # Project concatenated encodings down to embedding_dim
            pos_encodings = self.pos_encoding_proj(pos_encodings)
        else:
            # Fourier-only mode
            pos_encodings = learned_pos_encodings

        x = atom_embeddings + pos_encodings
        x = self.input_norm(x)

        for block in self.attention_blocks:
            x = block(x, mask=masks, deterministic=not training, rngs=rngs)

        # Mean pooling over atoms (excluding padding)
        x = x * masks[:, :, None]
        denom = jnp.maximum(jnp.sum(masks, axis=1, keepdims=True), 1e-6)
        x = jnp.sum(x, axis=1) / denom
        
        return self.final_ff(x, training=training)
