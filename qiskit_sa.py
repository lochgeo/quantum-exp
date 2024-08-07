import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.providers.basic_provider import BasicProvider

np.random.seed(42)

def quantum_scaled_dot_product_attention(q, k, v, d_k):
    q = q[0, 0]
    k = k[0, 0]
    v = v[0, 0]
    
    print("Quantum Q:", q)
    print("Quantum K:", k)
    print("Quantum V:", v)
    
    n_qubits = int(np.ceil(np.log2(d_k)))
    
    q_reg = QuantumRegister(n_qubits, 'q')
    k_reg = QuantumRegister(n_qubits, 'k')
    v_reg = QuantumRegister(n_qubits, 'v')
    anc_reg = QuantumRegister(1, 'ancilla')
    c_reg = ClassicalRegister(n_qubits, 'output')
    
    qc = QuantumCircuit(q_reg, k_reg, v_reg, anc_reg, c_reg)
    
    # Encode q and k into quantum states
    for i in range(n_qubits):
        if q[i] > 0:
            qc.ry(2 * np.arcsin(np.sqrt(np.abs(q[i]))), q_reg[i])
        if k[i] > 0:
            qc.ry(2 * np.arcsin(np.sqrt(np.abs(k[i]))), k_reg[i])
    
    # Apply QFT to approximate dot product
    qc.append(QFT(n_qubits), q_reg[:])
    qc.append(QFT(n_qubits), k_reg[:])
    
    # Use the ancilla qubit to control the "multiplication"
    for i in range(n_qubits):
        qc.ccx(q_reg[i], k_reg[i], anc_reg[0])
    
    # Apply inverse QFT to get the result
    qc.append(QFT(n_qubits).inverse(), q_reg[:])
    
    # Apply scaling factor (1/sqrt(d_k))
    scaling_angle = 2 * np.arcsin(1 / np.sqrt(d_k))
    qc.ry(scaling_angle, anc_reg[0])
    
    # Apply "softmax" (this is a very rough approximation)
    qc.ry(np.pi/4, anc_reg[0])  # Apply a rotation to simulate activation
    
    # Apply the result to v (again, a simplification)
    for i in range(n_qubits):
        qc.cry(2 * np.arcsin(np.sqrt(np.abs(v[i]))), anc_reg[0], v_reg[i])
    
    # Measure the output
    qc.measure(v_reg, c_reg)
    
    return qc

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

qc = quantum_scaled_dot_product_attention(q, k, v, d_k)

# Run the quantum circuit on a simulator
backend = BasicProvider().get_backend('basic_simulator')
nqc = transpile(qc, backend)
job = backend.run(nqc)
result = job.result()

# Get the measurement results
counts = result.get_counts(qc)
print("Quantum output (measurement counts):", counts)

# Calculate the expectation value
expectation = sum([int(key, 2) * value for key, value in counts.items()]) / 1000
print("Quantum output (expectation value):", expectation)
