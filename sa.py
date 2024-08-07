import numpy as np

np.random.seed(42)

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=1, keepdims=True)

def scaled_dot_product_attention(q, k, v, d_k):
    print("Classical Q:", q)
    print("Classical K:", k)
    print("Classical V:", v)
    
    attn_scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
    print("Classical attn_scores:", attn_scores)
    
    attn_weights = softmax(attn_scores)
    print("Classical attn_weights:", attn_weights)
    
    output = np.matmul(attn_weights, v)
    return output

# Example usage
batch_size = 1
seq_len = 1
d_k = 8
d_v = 8

q = np.random.randn(batch_size, seq_len, d_k)
k = np.random.randn(batch_size, seq_len, d_k)
v = np.random.randn(batch_size, seq_len, d_v)

# Normalize the vectors
q = q / np.linalg.norm(q)
k = k / np.linalg.norm(k)
v = v / np.linalg.norm(v)

output = scaled_dot_product_attention(q, k, v, d_k)
print("Classical output:", output)
