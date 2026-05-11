"""
Transformers Module - Self-Attention and Transformer Block from Scratch.

Implements the core components of the Transformer architecture using only NumPy:
  - Scaled Dot-Product Attention
  - Multi-Head Self-Attention
  - Positional Encoding (sinusoidal)
  - Layer Normalization
  - Feed-Forward Network
  - Transformer Block (combining all components)

This is a forward-only implementation for educational purposes. It demonstrates
the mathematical operations behind Transformers without requiring a deep
learning framework.

Example:
    >>> from transformers import TransformerBlock
    >>> block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
    >>> output = block.forward(input_embeddings)
"""

import numpy as np


class LayerNormalization:
    """Layer Normalization as described in "Layer Normalization" (Ba et al., 2016).

    Normalizes the input across the feature dimension to stabilize training.
    Supports learnable scale (gamma) and shift (beta) parameters.

    Attributes:
        gamma: Scale parameter (learnable), initialized to ones.
        beta: Shift parameter (learnable), initialized to zeros.
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        """Initialize Layer Normalization.

        Args:
            d_model: Dimension of the input features.
            epsilon: Small constant added to variance for numerical stability.
        """
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))
        self.epsilon = epsilon

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization.

        Normalizes each sample independently: x_norm = gamma * (x - mu) / sqrt(var + eps) + beta

        Args:
            x: Input of shape (batch_size, seq_len, d_model) or (batch_size, d_model).

        Returns:
            Normalized output of the same shape as input.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta


class PositionalEncoding:
    """Sinusoidal Positional Encoding as described in "Attention Is All You Need".

    Generates position embeddings using sine and cosine functions of different
    frequencies to inject positional information into the input sequence.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attributes:
        pe: Precomputed positional encoding matrix of shape (max_len, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_prob: float = 0.1):
        """Initialize Positional Encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length supported.
            dropout_prob: Probability of zeroing elements (not implemented here
                since this is a forward-only demo).
        """
        self.d_model = d_model
        self.max_len = max_len

        # Precompute the positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)  # (max_len, 1)

        # Compute the denominator: 10000^(2i/d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2).astype(float) * -(np.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = np.sin(position * div_term)  # Even indices
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices

        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Embeddings with positional information of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class ScaledDotProductAttention:
    """Scaled Dot-Product Attention mechanism.

    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Attributes:
        d_k: Dimension of the keys (and queries).
    """

    def __init__(self, d_k: int):
        """Initialize Scaled Dot-Product Attention.

        Args:
            d_k: Dimension of key vectors.
        """
        self.d_k = d_k

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray = None,
    ) -> tuple:
        """Compute scaled dot-product attention.

        Args:
            query: Query matrix of shape (batch_size, n_heads, seq_len, d_k).
            key: Key matrix of shape (batch_size, n_heads, seq_len, d_k).
            value: Value matrix of shape (batch_size, n_heads, seq_len, d_k).
            mask: Optional mask of shape (batch_size, 1, 1, seq_len) or
                  (batch_size, 1, seq_len, seq_len). Positions with True/1
                  are masked (set to -inf before softmax).

        Returns:
            Tuple of (output, attention_weights):
                - output: Attention output of shape (batch_size, n_heads, seq_len, d_k).
                - attention_weights: Softmax weights of shape
                  (batch_size, n_heads, seq_len, seq_len).
        """
        # Compute attention scores: Q @ K^T
        # Shapes: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
        #       -> (batch, heads, seq_q, seq_k)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax over the key dimension (last axis)
        attention_weights = self._softmax(scores, axis=-1)

        # Apply attention weights to values
        output = np.matmul(attention_weights, value)

        return output, attention_weights

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax function.

        Args:
            x: Input array.
            axis: Axis along which to compute softmax.

        Returns:
            Softmax output with values in (0, 1) summing to 1 along the axis.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """Multi-Head Self-Attention mechanism.

    Projects Q, K, V into multiple subspaces, applies parallel attention,
    then concatenates and projects the results.

    Attention(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    Attributes:
        n_heads: Number of attention heads.
        d_model: Total model dimension.
        d_k: Dimension per head (d_model // n_heads).
        W_Q, W_K, W_V: Projection weight matrices for Q, K, V.
        W_O: Output projection weight matrix.
    """

    def __init__(self, d_model: int, n_heads: int):
        """Initialize Multi-Head Attention.

        Args:
            d_model: Model dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize projection matrices with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        rng = np.random.default_rng(42)

        self.W_Q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_K = rng.normal(0, scale, size=(d_model, d_model))
        self.W_V = rng.normal(0, scale, size=(d_model, d_model))
        self.W_O = rng.normal(0, scale, size=(d_model, d_model))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray = None,
    ) -> tuple:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights):
                - output: Transformed output of shape (batch_size, seq_len, d_model).
                - attention_weights: Per-head attention weights of shape
                  (batch_size, n_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = x @ self.W_Q  # (batch, seq_len, d_model)
        K = x @ self.W_K
        V = x @ self.W_V

        # Reshape for multi-head: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attention.forward(Q, K, V, mask)

        # Concatenate heads: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = attn_output @ self.W_O

        return output, attn_weights


class FeedForward:
    """Position-wise Feed-Forward Network.

    Two linear transformations with GELU activation in between:
        FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

    This uses ReLU for simplicity; the original paper uses GELU.

    Attributes:
        W1, b1: First linear layer weights and bias.
        W2, b2: Second linear layer weights and bias.
    """

    def __init__(self, d_model: int, d_ff: int):
        """Initialize the Feed-Forward Network.

        Args:
            d_model: Input and output dimension.
            d_ff: Hidden layer dimension (typically 4x d_model).
        """
        # He initialization for weights
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        rng = np.random.default_rng(42)

        self.W1 = rng.normal(0, scale1, size=(d_model, d_ff))
        self.b1 = np.zeros((1, d_ff))
        self.W2 = rng.normal(0, scale2, size=(d_ff, d_model))
        self.b2 = np.zeros((1, d_model))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation function (Gaussian Error Linear Unit).

        GELU(x) = x * Phi(x) where Phi is the CDF of the standard normal.
        Uses the tanh approximation for numerical stability.
        """
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the feed-forward network.

        Args:
            x: Input of shape (batch_size, seq_len, d_model).

        Returns:
            Output of shape (batch_size, seq_len, d_model).
        """
        hidden = x @ self.W1 + self.b1
        hidden = self._gelu(hidden)  # Could also use _relu
        output = hidden @ self.W2 + self.b2
        return output


class TransformerBlock:
    """Complete Transformer Block combining all components.

    Architecture:
        1. Multi-Head Self-Attention
        2. Residual Connection + Layer Normalization
        3. Feed-Forward Network
        4. Residual Connection + Layer Normalization

    This matches the "Pre-LN" variant used in modern implementations:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Attributes:
        mha: MultiHeadAttention instance.
        ff: FeedForward instance.
        norm1: LayerNormalization for attention.
        norm2: LayerNormalization for feed-forward.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        """Initialize the Transformer Block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
        """
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = LayerNormalization(d_model=d_model)
        self.norm2 = LayerNormalization(d_model=d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Apply the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention with residual connection and layer normalization
        x_norm = self.norm1.forward(x)
        attn_output, attn_weights = self.mha.forward(x_norm, mask)
        x = x + attn_output

        # Feed-forward with residual connection and layer normalization
        x_norm = self.norm2.forward(x)
        ff_output = self.ff.forward(x_norm)
        x = x + ff_output

        return x


class SimpleTransformer:
    """A simple stack of Transformer blocks for sequence processing.

    Attributes:
        pos_encoding: PositionalEncoding instance.
        blocks: List of TransformerBlock instances.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_len: int = 100,
    ):
        """Initialize the Simple Transformer.

        Args:
            vocab_size: Size of the vocabulary (for embedding layer).
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            n_layers: Number of transformer blocks to stack.
            max_len: Maximum sequence length.
        """
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding (randomly initialized for demo)
        rng = np.random.default_rng(42)
        self.token_embedding = rng.normal(0, 0.02, size=(vocab_size, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ]

        # Final layer normalization
        self.norm = LayerNormalization(d_model=d_model)

        # Output projection (d_model -> vocab_size)
        self.output_proj = rng.normal(0, 0.02, size=(d_model, vocab_size))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the full transformer.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings + scaling
        x = self.token_embedding[token_ids] * np.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final normalization and projection
        x = self.norm.forward(x)
        logits = x @ self.output_proj

        return logits

    def generate(self, token_ids: np.ndarray, max_new_tokens: int = 10) -> np.ndarray:
        """Generate new tokens autoregressively (greedy decoding).

        Args:
            token_ids: Starting token IDs of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Extended token ID sequence of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(token_ids)

            # Get logits for last position and apply greedy decoding
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)  # (batch_size, 1)

            # Append to sequence
            token_ids = np.concatenate([token_ids, next_token], axis=1)

        return token_ids


if __name__ == "__main__":
    print("=" * 60)
    print("Transformers Module - Component Demos")
    print("=" * 60)

    # --- 1. Positional Encoding ---
    print("\n1. Positional Encoding")
    print("-" * 40)
    pe = PositionalEncoding(d_model=16, max_len=20)
    print(f"  Positional encoding shape: {pe.pe.shape}")
    print(f"  First position encoding:   {pe.pe[0, :8].round(4)}")
    print(f"  Second position encoding:  {pe.pe[1, :8].round(4)}")

    # --- 2. Layer Normalization ---
    print("\n2. Layer Normalization")
    print("-" * 40)
    ln = LayerNormalization(d_model=4)
    x_test = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    x_norm = ln.forward(x_test)
    print(f"  Input:\n{x_test}")
    print(f"  Normalized:\n{x_norm.round(4)}")
    print(f"  Means per row: {x_norm.mean(axis=1).round(6)}")
    print(f"  Vars per row:  {x_norm.var(axis=1).round(6)}")

    # --- 3. Scaled Dot-Product Attention ---
    print("\n3. Scaled Dot-Product Attention")
    print("-" * 40)
    sdpa = ScaledDotProductAttention(d_k=8)
    batch, heads, seq_len, d_k = 2, 4, 5, 8
    Q = np.random.randn(batch, heads, seq_len, d_k)
    K = np.random.randn(batch, heads, seq_len, d_k)
    V = np.random.randn(batch, heads, seq_len, d_k)
    attn_out, attn_w = sdpa.forward(Q, K, V)
    print(f"  Query shape:   {Q.shape}")
    print(f"  Key shape:     {K.shape}")
    print(f"  Value shape:   {V.shape}")
    print(f"  Output shape:  {attn_out.shape}")
    print(f"  Weights shape: {attn_w.shape}")
    print(f"  Attention weights sum (should be ~1.0): {attn_w.sum(axis=-1).mean():.6f}")

    # --- 4. Multi-Head Attention ---
    print("\n4. Multi-Head Self-Attention")
    print("-" * 40)
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    x_mha = np.random.randn(3, 10, 64)
    mha_out, mha_w = mha.forward(x_mha)
    print(f"  Input shape:       {x_mha.shape}")
    print(f"  Output shape:      {mha_out.shape}")
    print(f"  Weights shape:     {mha_w.shape}")
    print(f"  Output mean: {mha_out.mean():.4f}, std: {mha_out.std():.4f}")

    # --- 5. Feed-Forward Network ---
    print("\n5. Feed-Forward Network")
    print("-" * 40)
    ff = FeedForward(d_model=64, d_ff=256)
    x_ff = np.random.randn(3, 10, 64)
    ff_out = ff.forward(x_ff)
    print(f"  Input shape:  {x_ff.shape}")
    print(f"  Output shape: {ff_out.shape}")
    print(f"  Output mean: {ff_out.mean():.4f}, std: {ff_out.std():.4f}")

    # --- 6. Transformer Block ---
    print("\n6. Full Transformer Block")
    print("-" * 40)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    x_block = np.random.randn(2, 8, 64)
    block_out = block.forward(x_block)
    print(f"  Input shape:  {x_block.shape}")
    print(f"  Output shape: {block_out.shape}")
    print(f"  Output mean: {block_out.mean():.4f}, std: {block_out.std():.4f}")

    # --- 7. Full Transformer ---
    print("\n7. Simple Transformer (autoregressive generation)")
    print("-" * 40)
    transformer = SimpleTransformer(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        max_len=50,
    )

    # Create a simple input sequence
    input_tokens = np.array([[1, 5, 10, 15, 20], [3, 8, 12, 18, 25]])
    print(f"  Input tokens shape: {input_tokens.shape}")
    print(f"  Input tokens:\n{input_tokens}")

    # Forward pass
    logits = transformer.forward(input_tokens)
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Predicted next token (greedy) for sequence 0: {np.argmax(logits[0, -1, :])}")
    print(f"  Predicted next token (greedy) for sequence 1: {np.argmax(logits[1, -1, :])}")

    # Generate tokens
    generated = transformer.generate(input_tokens, max_new_tokens=5)
    print(f"  Generated sequence shape: {generated.shape}")
    print(f"  Full generated sequence 0: {generated[0]}")
    print(f"  Full generated sequence 1: {generated[1]}")
