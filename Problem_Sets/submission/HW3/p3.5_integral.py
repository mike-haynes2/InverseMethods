import numpy as np
import scipy

# Column 1: y values
y = np.array([
    0.0250, 0.0750, 0.1250, 0.1750, 0.2250, 0.2750, 0.3250, 0.3750, 0.4250, 0.4750,
    0.5250, 0.5750, 0.6250, 0.6750, 0.7250, 0.7750, 0.8250, 0.8750, 0.9250, 0.9750
])

# Column 2: d(y) values
d_y = np.array([
    0.2388, 0.2319, 0.2252, 0.2188, 0.2126, 0.2066, 0.2008, 0.1952, 0.1898, 0.1846,
    0.1795, 0.1746, 0.1699, 0.1654, 0.1610, 0.1567, 0.1526, 0.1486, 0.1447, 0.1410
])

# N = 20 based on Table 3.2
N = len(y)
dx = 1.0 / N
x_edges = np.linspace(0, 1, N + 1)

# Function to compute the integral G[i,j] analytically
def kernel_integral(y, x_start, x_end):
    # Handle the y=0 case if necessary (though Table 3.2 starts at 0.025)
    if y == 0:
        return 0.5 * (x_end**2 - x_start**2)
    
    def indef_int(val):
        return - (val * np.exp(-y * val) / y) - (np.exp(-y * val) / (y**2))
    
    return indef_int(x_end) - indef_int(x_start)

# Initialize and fill G matrix
G = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        G[i, j] = kernel_integral(y[i], x_edges[j], x_edges[j+1])

# Solve the system Gm = d
# Note: This system is ill-conditioned (Fredholm IFK), so simple inv is risky.
m_sol = np.linalg.solve(G, d_y)
m_dag = scipy.linalg.pinv(G) @ d_y

print(f"G Matrix Shape: {G.shape}")
print(f"Condition Number: {np.linalg.cond(G):.2e}")

print(m_sol,m_dag)


U, s, Vh = np.linalg.svd(G)
Prats = np.abs(U.T @ d_y) / s

import matplotlib.pyplot as plt

plt.scatter(np.arange(len(Prats)),Prats, marker='o', color='navy')
plt.title('Picard Ratios', fontweight='bold')
plt.yscale('log')
plt.grid()
plt.xlabel('index n')
plt.ylabel('ratio')
plt.show()
# plt.savefig('p3.5c_Picard.png', dpi=250)
# plt.close()



def tsvd(A, k=None, full_matrices=False):
    """
    Custom wrapper for Truncated SVD.
    Returns U, S (as 1D array), and Vh truncated to k components.
    """
    # 1. Perform standard SVD
    U, s, Vh = scipy.linalg.svd(A, full_matrices=full_matrices)
    
    # 2. Determine truncation point
    # If k is None, keep all components
    if k is None:
        k = len(s)
    
    # 3. Truncate components
    Uk = U[:, :k]   # Keep first k columns
    sk = s[:k]      # Keep first k singular values
    Vhk = Vh[:k, :] # Keep first k rows (since Vh is V-Hermitian)
    
    return (Vhk.T @ (np.linalg.inv(np.diag(sk)) @ Uk.T))


Gdagg = tsvd(G, k=4, full_matrices=True)
mdagg = Gdagg @ d_y
plt.plot(y, mdagg, color='teal')
plt.title('Recovered model $m(x)$')
plt.xlabel('$y$')
plt.ylabel('$m$')
plt.grid()
plt.show()
#plt.savefig('p5c_m.png')

#print(f"Condition Number: {np.linalg.cond(np.linalg.inv(Gdagg)):.2e}")