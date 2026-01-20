import flax.linen as nn
import jax.numpy as jnp


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.
    
    Args:
        config: Dictionary containing embedding_dim, num_heads, dropout_rate.
    """
    config: dict

    @nn.compact
    def __call__(self, q, k, v, mask=None, deterministic=False, rngs=None):
        d_model = self.config['embedding_dim']
        num_heads = self.config['num_heads']
        d_head = d_model // num_heads
        dropout_rate = self.config.get('dropout_rate', 0.0)

        # Linear projections for Q, K, V
        q_proj = nn.Dense(d_model, use_bias=False, name='q_proj')(q)
        k_proj = nn.Dense(d_model, use_bias=False, name='k_proj')(k)
        v_proj = nn.Dense(d_model, use_bias=False, name='v_proj')(v)

        # Split into heads: (batch, seq, d_model) -> (batch, heads, seq, d_head)
        def split_heads(x):
            return x.reshape(x.shape[:-1] + (num_heads, d_head)).transpose((0, 2, 1, 3))
        
        q_s, k_s, v_s = map(split_heads, (q_proj, k_proj, v_proj))

        # Scaled dot-product attention
        scale = d_head ** -0.5
        attention_logits = jnp.einsum('bhqd,bhkd->bhqk', q_s, k_s) * scale
        
        # Apply padding mask
        if mask is not None:
            key_mask = mask[:, None, None, :]
            query_mask = mask[:, None, :, None]
            attention_logits = jnp.where(key_mask, attention_logits, -jnp.inf)
        else:
            query_mask = None

        # Avoid NaNs when a query has no valid keys
        if mask is not None:
            has_any_key = jnp.any(jnp.isfinite(attention_logits), axis=-1, keepdims=True)
            attention_logits = jnp.where(has_any_key, attention_logits, 0.0)

        # Softmax over keys
        attention = nn.softmax(attention_logits, axis=-1)
        
        # Zero out attention for padded queries
        if query_mask is not None:
            attention = attention * query_mask
            
        attn_dropout_rng = self.make_rng('dropout') if not deterministic else None
        attention = nn.Dropout(rate=dropout_rate)(attention, deterministic=deterministic, rng=attn_dropout_rng)
        
        # Combine heads: (batch, heads, seq, d_head) -> (batch, seq, d_model)
        output = jnp.einsum('bhqk,bhkd->bhqd', attention, v_s)
        output = output.transpose((0, 2, 1, 3)).reshape(output.shape[0], -1, d_model)

        # Output projection
        output = nn.Dense(d_model, use_bias=False, name='output_proj')(output)
        out_dropout_rng = self.make_rng('dropout') if not deterministic else None
        return nn.Dropout(rate=dropout_rate)(output, deterministic=deterministic, rng=out_dropout_rng)


class TransformerMLP(nn.Module):
    """Feed-forward MLP used within transformer blocks.
    
    Args:
        config: Dictionary containing embedding_dim, ff_dim, dropout_rate.
    """
    config: dict

    @nn.compact
    def __call__(self, x, deterministic=False, rngs=None):
        d_model = self.config['embedding_dim']
        d_proj = self.config['ff_dim']
        dropout_rate = self.config.get('dropout_rate', 0.0)
        
        x = nn.Dense(d_proj)(x)
        x = nn.silu(x)
        mlp_dropout_rng1 = self.make_rng('dropout') if not deterministic else None
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=deterministic, rng=mlp_dropout_rng1)
        x = nn.Dense(d_model)(x)
        mlp_dropout_rng2 = self.make_rng('dropout') if not deterministic else None
        return nn.Dropout(rate=dropout_rate)(x, deterministic=deterministic, rng=mlp_dropout_rng2)


class AttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block with self-attention and feed-forward layers.
    
    Args:
        config: Dictionary containing model hyperparameters.
    """
    config: dict

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False, rngs=None):
        # Pre-LN self-attention
        y = nn.LayerNorm(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(self.config)(
            y, y, y, mask=mask, deterministic=deterministic, rngs=rngs
        )
        x = x + attention_output

        # Pre-LN feed-forward
        y = nn.LayerNorm(epsilon=1e-6)(x)
        ff_output = TransformerMLP(self.config)(y, deterministic=deterministic, rngs=rngs)
        x = x + ff_output
        
        # Ensure padded positions remain zeroed
        if mask is not None:
            x = x * mask[:, :, None]
        return x
