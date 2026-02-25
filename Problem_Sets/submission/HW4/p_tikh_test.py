import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup the True Model and Forward Matrix
# ==========================================
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)

# Our "true" underlying vector m (a smooth wave)
m_true = np.sin(x) + np.exp(-(x-5)**2)

# Our Forward Operator G (a Gaussian blurring/smoothing matrix)
# Multiplying by G will smooth out any signal.
kernel = np.exp(-0.5 * (np.arange(n) - n//2)**2 / 4.0)
G = np.array([np.roll(kernel, i - n//2) for i in range(n)]) / np.sum(kernel)

# ==========================================
# 2. Generate Noisy Data
# ==========================================
d_exact = G @ m_true
noise_level = 0.05
noise = np.random.normal(0, noise_level, n)
d_noisy = d_exact + noise

# ==========================================
# 3. Naive Inversion (Standard Least Squares)
# ==========================================
# m_naive = (G^T G)^-1 G^T d
# This will likely blow up because G is ill-conditioned and d has noise.
try:
    m_naive = np.linalg.solve(G.T @ G, G.T @ d_noisy)
except np.linalg.LinAlgError:
    m_naive = np.zeros(n) # Fallback if perfectly singular

# ==========================================
# 4. Tikhonov Regularization (First-Order)
# ==========================================
alpha = 0.5  # Regularization parameter (controls amount of smoothing)

# Create L: First derivative matrix (penalizes roughness)
# It computes adjacent differences: m_i - m_{i+1}
L = np.eye(n) - np.diag(np.ones(n-1), k=1)
L[-1, -1] = 0 # Boundary condition adjustment

# Tikhonov equation: m_tik = (G^T G + alpha^2 L^T L)^-1 G^T d
G_T_G = G.T @ G
L_T_L = L.T @ L
G_T_d = G.T @ d_noisy

m_tikhonov = np.linalg.solve(G_T_G + (alpha**2) * L_T_L, G_T_d)

# ==========================================
# 5. Visualization
# ==========================================
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title("Forward Problem: True Model vs Noisy Data")
plt.plot(x, m_true, 'k-', linewidth=2, label="True Model (m)")
plt.plot(x, d_noisy, 'ro', markersize=4, label="Noisy Data (d = Gm + noise)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Inverse Problem: Recovering the Model")
plt.plot(x, m_true, 'k-', linewidth=2, label="True Model")
plt.plot(x, m_naive, 'gray', alpha=0.5, label="Naive Inversion (Blows up!)")
plt.plot(x, m_tikhonov, 'b-', linewidth=2, label=f"Tikhonov Inversion (alpha={alpha})")
plt.ylim([-2, 3]) # Restricting Y-axis because naive inversion values are massive
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()