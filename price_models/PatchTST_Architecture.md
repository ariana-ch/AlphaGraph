# PatchTST (Patch Time Series Transformer) Architecture

## Overview

PatchTST is a state-of-the-art time series forecasting model that combines patch-based processing with transformer architecture. It was introduced in the paper ["A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"](https://arxiv.org/pdf/2211.14730.pdf).

## Key Innovation: Patch-Based Processing

Instead of processing individual time steps, PatchTST divides the time series into **patches** (contiguous subsequences), treating each patch as a "token" similar to how Vision Transformers treat image patches. This approach:

- **Reduces sequence length**: Makes transformers more efficient for long sequences
- **Captures local patterns**: Each patch preserves local temporal dependencies
- **Enables better attention**: Attention operates on meaningful chunks rather than individual points

## Architecture Components

### 1. Input Processing Pipeline

```
Raw Time Series → Entry Layer → Normalization → Patch Embedding → Transformer Encoder → Prediction Head
```

### 2. Detailed Architecture Flow

#### **Step 1: Entry Layer**
- Maps original features to a reduced feature space
- `Linear(original_features → enc_in)`
- Reduces computational complexity

#### **Step 2: Time Series Normalization**
- **Instance Normalization**: Normalizes each time series individually
- Stores means and standard deviations for denormalization
- Helps with non-stationary time series

#### **Step 3: Patch Embedding**
- **Patching**: Divides sequence into overlapping patches
- **Value Embedding**: Projects each patch to d_model dimensions
- **Positional Embedding**: Adds position information to patches
- **Dropout**: Regularization

#### **Step 4: Transformer Encoder**
- Multi-head self-attention layers
- Feed-forward networks
- Layer normalization and residual connections
- Batch normalization for stability

#### **Step 5: Prediction Head**
- **Reshape**: Back to [batch, variables, features, patches]
- **Flatten**: Combines feature and patch dimensions
- **Linear Projection**: Maps to prediction length
- **Denormalization**: Restores original scale

## Input Parameters Explained

### **Core Architecture Parameters**

#### `lookback_len` (default: 30)
- **Purpose**: Length of input time series sequence
- **Example**: 30 means using last 30 time steps to predict future
- **Impact**: Longer sequences capture more historical patterns but increase computation

#### `pred_len` (default: 1)
- **Purpose**: Number of future time steps to predict
- **Example**: 1 means predicting next time step
- **Impact**: Longer predictions are generally harder but more useful for planning

#### `d_model` (default: 128)
- **Purpose**: Hidden dimension size throughout the transformer
- **Impact**: 
  - Larger = more model capacity but slower training
  - Must be divisible by `n_heads`
- **Typical range**: 64-512

#### `n_heads` (default: 4)
- **Purpose**: Number of attention heads in multi-head attention
- **Impact**:
  - More heads = model can attend to different aspects simultaneously
  - `d_model` must be divisible by `n_heads`
- **Typical range**: 4-16

#### `d_ff` (default: 256)
- **Purpose**: Hidden dimension in feed-forward networks
- **Rule of thumb**: Usually 2-4x `d_model`
- **Impact**: Larger = more non-linear capacity but more parameters

#### `dropout` (default: 0.1)
- **Purpose**: Dropout rate for regularization
- **Applied to**: Patch embedding, attention, feed-forward layers
- **Impact**: Higher dropout reduces overfitting but may hurt performance

#### `e_layers` (default: 2)
- **Purpose**: Number of transformer encoder layers
- **Impact**: 
  - More layers = deeper model with more capacity
  - Diminishing returns beyond 4-6 layers for most time series
- **Typical range**: 2-6

### **Attention Parameters**

#### `factor` (default: 5)
- **Purpose**: Controls attention sparsity (used in some attention variants)
- **Impact**: Higher factor = sparser attention = faster computation
- **Note**: In full attention (current implementation), this has minimal effect

#### `activation` (default: 'gelu')
- **Purpose**: Activation function in feed-forward networks
- **Options**: 'relu', 'gelu'
- **Impact**: GELU generally works better for transformers

### **Feature Dimensions**

#### `features` (default: 45)
- **Purpose**: Number of input features per time step
- **Example**: In financial data, might include price, volume, technical indicators
- **Impact**: More features provide richer information but increase computation

#### `enc_in` (default: 20)
- **Purpose**: Reduced feature dimension after entry layer
- **Impact**: 
  - Compression of original features
  - Reduces computational load
  - Should capture essential information

### **Patch Parameters**

#### `patch_len` (default: 10)
- **Purpose**: Length of each patch
- **Impact**:
  - Larger patches = fewer patches = faster attention but less granular
  - Smaller patches = more patches = slower but more detailed
- **Rule**: Should divide `lookback_len` reasonably well

#### `stride` (default: 5)
- **Purpose**: Step size between consecutive patches
- **Impact**:
  - `stride = patch_len`: Non-overlapping patches
  - `stride < patch_len`: Overlapping patches (more common)
  - Smaller stride = more patches = richer representation

## How Patching Works

### Example with `lookback_len=30`, `patch_len=10`, `stride=5`:

```
Original sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 30]

Patches:
Patch 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]     (positions 0-9)
Patch 2: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (positions 5-14)
Patch 3: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] (positions 10-19)
...

Number of patches = (30 - 10) / 5 + 1 + 1 = 6 patches
```

### Patch Calculation Formula:
```python
patch_num = (lookback_len - patch_len) // stride + 1 + 1
```

## Memory and Computational Complexity

### **Attention Complexity**
- **Original Transformer**: O(L²) where L = sequence length
- **PatchTST**: O(P²) where P = number of patches
- **Reduction**: P << L, so much more efficient

### **Memory Usage**
- Scales with: `batch_size × n_vars × patch_num × d_model`
- Patches significantly reduce memory compared to full sequence attention

## Best Practices for Parameter Selection

### **For Financial Time Series (like your use case):**

```python
# Conservative (faster, good baseline)
PatchTST(
    lookback_len=30,      # 1 month of daily data
    pred_len=1,           # Next day prediction
    d_model=64,           # Moderate capacity
    n_heads=4,            # Standard
    d_ff=128,             # 2x d_model
    dropout=0.1,          # Standard regularization
    e_layers=2,           # Simple model
    patch_len=10,         # ~2 week patches
    stride=5              # 50% overlap
)

# Aggressive (slower, more capacity)
PatchTST(
    lookback_len=60,      # 2 months of daily data
    pred_len=5,           # 1 week prediction
    d_model=128,          # Higher capacity
    n_heads=8,            # More attention heads
    d_ff=256,             # 2x d_model
    dropout=0.15,         # More regularization
    e_layers=4,           # Deeper model
    patch_len=12,         # ~2.5 week patches
    stride=6              # 50% overlap
)
```

### **Parameter Interaction Guidelines:**

1. **d_model must be divisible by n_heads**
2. **patch_len should be meaningful for your domain** (e.g., weekly patterns = 5-7 days)
3. **stride typically 50-75% of patch_len** for good overlap
4. **d_ff usually 2-4x d_model**
5. **More layers help with complex patterns but risk overfitting**

## Model Output

The model outputs a tensor of shape `[batch_size, pred_len]` representing predicted values for each prediction time step. In portfolio optimization, these are typically interpreted as:

- **Raw scores**: Unnormalized portfolio weights
- **Processed via softmax/entmax**: Normalized portfolio weights that sum to 1

## Advantages of PatchTST

1. **Efficiency**: O(P²) vs O(L²) attention complexity
2. **Local Pattern Capture**: Patches preserve local temporal structure
3. **Scalability**: Handles long sequences better than vanilla transformers
4. **Flexibility**: Works well across different time series domains
5. **State-of-the-art**: Competitive performance on many benchmarks

## Limitations

1. **Patch Selection**: Requires careful tuning of patch_len and stride
2. **Information Loss**: Patching may lose some fine-grained temporal details
3. **Memory**: Still requires significant memory for large models
4. **Complexity**: More hyperparameters to tune compared simpler models 