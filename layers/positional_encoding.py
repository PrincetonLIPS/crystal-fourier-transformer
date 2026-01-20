import flax.linen as nn
import jax
import jax.numpy as jnp

def create_encoding_vector(pos, recip_matrix, embedding_dim):
    encoding = []
    for i in range(0, embedding_dim, 6):  # 6 values per dimension (sin and cos for x, y, z)
        freq_vector = jnp.array([
            1.0 / (10000 ** ((i + 2*dim) / embedding_dim)) for dim in range(3)
        ])
        scaled_freq = jnp.dot(recip_matrix, freq_vector)
        
        for dim in range(3):
            encoding.append(jnp.sin(pos[dim] * scaled_freq[dim]))
            encoding.append(jnp.cos(pos[dim] * scaled_freq[dim]))
    
    return jnp.array(encoding)[:embedding_dim]

class SpaceGroupDenseLayer(nn.Module):
    features: int
    num_space_groups: int = 230

    @nn.compact
    def __call__(self, inputs, space_group):
        batch_size, num_atoms, input_dim = inputs.shape
        weights = self.param('weights', 
                             lambda rng, shape: jax.random.normal(rng, shape, dtype=jnp.complex64),
                             (self.num_space_groups, input_dim, self.features))
        biases = self.param('bias', 
                            lambda rng, shape: jax.random.normal(rng, shape, dtype=jnp.complex64),
                            (self.num_space_groups, self.features))

        # Index into the weights and biases for the given space group
        selected_weights = weights[space_group]
        selected_biases = biases[space_group]

        outputs = jnp.einsum('bni,bif->bnf', inputs, selected_weights)
        outputs = outputs + selected_biases[:, None, :]
        
        return outputs
    

class NaivePositionalEncoding(nn.Module):
    config: dict

    def setup(self):
        self.dense1 = nn.Dense(features=self.config['encoding_hidden_dim'])
        self.dense2 = nn.Dense(features=self.config['embedding_dim'])

    @nn.compact
    def __call__(self, atom_positions, reciprocal_matrices, space_group):
        """
        Lattice-aware sines and cosines positional encoding.
        """
        embedding_dim = self.config['embedding_dim']
        vectorized_encoding = jax.vmap(jax.vmap(
                lambda pos, recip_matrix: create_encoding_vector(pos, recip_matrix, embedding_dim),
                in_axes=(0, None)
            ), in_axes=(0, 0))
        encoded_positions = vectorized_encoding(atom_positions, reciprocal_matrices)
        
        x = self.dense1(encoded_positions)
        x = jax.nn.silu(x)  
        x = self.dense2(x)
        
        return x

class PositionalEncoding(nn.Module):
    config: dict
    abc_combinations: jnp.ndarray
    graphs_array: jnp.ndarray

    def setup(self):
        self.norm = nn.LayerNorm()
        self.dense1 = SpaceGroupDenseLayer(features=self.config['encoding_hidden_dim'])
        self.dense2 = nn.Dense(features=self.config['embedding_dim'])
        self.recip_weights = nn.Dense(features=self.config['embedding_dim'])
    
    @nn.compact
    def __call__(self, atom_positions, reciprocal_matrices, space_group):
        """
        Fourier basis encoding for atom positions.
            
        Args:
            atom_positions (jnp.ndarray): Atom positions of shape (batch_size, num_atoms, 3).
            reciprocal_matrices (jnp.ndarray): Reciprocal matrices of shape (batch_size, 3, 3).
            space_group (jnp.ndarray): Space group numbers of shape (batch_size,).
            training (bool): Whether to use dropout.
            rngs (dict): A dictionary of PRNG keys for the Dropout layer.
        
        Returns:
            jnp.ndarray: Positional encoding.
        """

        # Compute reciprocal lattice points
        # transformed_abc = jax.vmap(lambda matrix: jnp.dot(self.abc_combinations, matrix))(reciprocal_matrices)
        # encodings = jax.vmap(lambda pos, t_abc: jnp.exp(1j * 2 * jnp.pi * jnp.dot(pos, t_abc.T)))(atom_positions, transformed_abc)

        encodings = jax.vmap(lambda pos: jnp.exp(1j * 2 * jnp.pi * jnp.dot(pos, self.abc_combinations.T)))(atom_positions)
        adjacency_matrices = self.graphs_array[space_group - 1]
        weighted_encodings = jax.vmap(jnp.matmul)(encodings, adjacency_matrices)

        # Trim to embedding dimension
        embedding_dim = self.config['embedding_dim']
        weighted_encodings = weighted_encodings[:, :, :embedding_dim]
        
        # Scale by reciprocal lattice embedding
        flat_recip = jax.vmap(lambda x: x.ravel(order='F'))(reciprocal_matrices)
        recip_weights = self.recip_weights(flat_recip)
        weighted_encodings = weighted_encodings * recip_weights[:, None, :]
        
        x = self.dense1(weighted_encodings, space_group - 1)
        x = jnp.concatenate([jnp.real(x), jnp.imag(x)], axis=-1)
        x = jax.nn.silu(x)
        x = self.dense2(x)
        return x