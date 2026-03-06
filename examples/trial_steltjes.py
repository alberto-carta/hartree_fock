#%%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def get_matsubara_interpolator(Z, M1, M3):
    """
    Fits Im(Sigma) on the Matsubara axis.
    Z: Quasiparticle weight matrix
    M1, M3: High-frequency moments (from Hubbard I)
    
    Constraint 1 (Slope at 0): Im(Sigma) ~ w_n * (I - Z^-1)
    Constraint 2 (Tail): Im(Sigma) ~ M1 / (i*w_n) + M3 / (i*w_n)^3
    """
    dim = Z.shape[0]
    I = np.eye(dim)
    
    # Target slope at zero
    # Im(Sigma) = w_n * Slope -> Slope = (I - Z^-1)
    Slope = I - np.linalg.inv(Z)
    
    # We use the form: Sigma(iw) = (iw * M1) / ( (iw)^2 + iw*Gamma + Omega^2 )
    # At high iw: Sigma ~ M1 / iw  (Matches M1)
    # At low iw: Sigma ~ (iw * M1) / Omega^2 -> Slope = M1 * Omega^-2
    
    # 1. Solve for Omega^2 using the slope and M1
    # Omega^2 = Slope^-1 * M1
    Omega2 = np.linalg.inv(Slope) @ M1
    
    # 2. Solve for Gamma using M3
    # Expanding the form to O(1/(iw)^3):
    # Sigma ~ M1/(iw) - (M1*Gamma)/(iw)^2 + (M1*Gamma^2 - M1*Omega^2)/(iw)^3
    # Matching the 1/(iw)^3 term to M3:
    # M3 = M1*Gamma^2 - M1*Omega^2  => Gamma^2 = M1^-1 * (M3 + M1*Omega2)
    
    Gamma2_mat = np.linalg.inv(M1) @ (M3 + M1 @ Omega2)
    # Use matrix square root for Gamma
    from scipy.linalg import sqrtm
    Gamma = sqrtm(Gamma2_mat)
    
    def sigma_matsubara(wn):
        # iw_n term
        iw = 1j * wn
        # Rational matrix form: num * inv(denom)
        num = iw * M1
        denom = (iw**2) * I + iw * Gamma + Omega2
        return num @ np.linalg.inv(denom)

    return sigma_matsubara

# Example execution:


Z_mat = np.array([[0.95, 0.00], [0.0, 0.95]])
M1_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
M3_mat = np.array([[0.01, 0.0], [0.0, 0.01]])

sigma_fn = get_matsubara_interpolator(Z_mat, M1_mat, M3_mat)

# Test at a low frequency
print("Slope check at wn=0.01:", np.imag(sigma_fn(0.01)) / 0.01)
print("Target Slope (I - Z^-1):", np.eye(2) - np.linalg.inv(Z_mat))


# plot sigma of omega

wn_vals = np.linspace(0.01, 100, 1000)
sigma_vals = np.array([sigma_fn(wn) for wn in wn_vals])
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(wn_vals, np.imag(sigma_vals[:, 0, 0]), label='Im(Sigma_11)')
plt.plot(wn_vals, np.imag(sigma_vals[:, 1, 1]), label='Im(Sigma_22)')
plt.xlim(0, 20)
plt.xlabel('Matsubara Frequency (wn)')
plt.ylabel('Im(Sigma)')
plt.title('Interpolated Im(Sigma) on Matsubara Axis')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(wn_vals, np.real(sigma_vals[:, 0, 0]), label='Re(Sigma_11)')
plt.plot(wn_vals, np.real(sigma_vals[:, 1, 1]), label='Re(Sigma_22)')
plt.xlabel('Matsubara Frequency (wn)')
plt.ylabel('Re(Sigma)')
plt.title('Interpolated Re(Sigma) on Matsubara Axis')
plt.xlim(0, 20)
plt.legend()