import flax.linen as nn
import jax.numpy as jnp

class ResNetBlock(nn.Module):
    """A ResNet block with two dense layers and a residual connection."""
    hidden_dim: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        # First dense layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # Second dense layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # Add residual connection
        # If input and output dimensions don't match, we need to project the residual
        if residual.shape[-1] != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim)(residual)
        
        return x + residual

class FeedForward(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.config['final_hidden_1'])(x)
        x = nn.silu(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(self.config['final_hidden_2'])(x)
        x = nn.silu(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        return nn.Dense(1)(x)
