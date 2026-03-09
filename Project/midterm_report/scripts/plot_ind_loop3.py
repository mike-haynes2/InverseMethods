import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import spherical_jn, spherical_yn
import cartopy.crs as ccrs
import os

# ==========================================
# 1. Numerically Stable Zimmer Engine
# ==========================================
def calculate_Q_complex(f, sigma, r1_km, r2_km, R_target_km):
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f
    # Using a small epsilon to avoid k=0 if sigma is 0
    k = np.sqrt(-1j * mu0 * (sigma + 1e-9) * omega)
    
    a, b = r2_km * 1000, r1_km * 1000
    ka, kb = k * a, k * b
    
    # helper for derivatives: [z * f_n(z)]' = f_n(z) + z * f_n'(z)
    def dj(z): return spherical_jn(1, z) + z * spherical_jn(1, z, derivative=True)
    def dy(z): return spherical_yn(1, z) + z * spherical_yn(1, z, derivative=True)

    j1a, y1a = spherical_jn(1, ka), spherical_yn(1, ka)
    j1b, y1b = spherical_jn(1, kb), spherical_yn(1, kb)
    dja, dya = dj(ka), dy(ka)
    djb, dyb = dj(kb), dy(kb)

    # Calculate alpha with a small denominator guard
    denom_alpha = dyb
    if np.abs(denom_alpha) < 1e-20: denom_alpha = 1e-20
    alpha = djb / denom_alpha
    
    # Calculate Q_surface
    num = (ka * j1a - dja) - alpha * (ka * y1a - dya)
    den = (ka * j1a + dja) - alpha * (ka * y1a + dya)
    
    if np.abs(den) < 1e-20: den = 1e-20
    Q_surface = num / den

    # Final check for NaN/Inf
    if not np.isfinite(Q_surface):
        return 0.0 + 0.0j # Default to no induction if math explodes

    return Q_surface * (a / (R_target_km * 1000))**3

# ==========================================
# 2. Refined Plotting Function
# ==========================================
def save_europa_figure(sigma_ocean, ice_thickness, ocean_thickness, alt_km, output_dir='figs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    R_E = 1560.0
    r2 = R_E - ice_thickness
    r1 = r2 - ocean_thickness
    R_obs = R_E + alt_km
    B0 = 210.0
    B_bg_xyz = np.array([0, B0, 0])
    freq = 1 / (11.1 * 3600)

    # 1. Math
    Q_val = calculate_Q_complex(freq, sigma_ocean, r1, r2, R_obs)
    
    res = 5
    lons = np.arange(-180, 180 + res, res)
    lats = np.arange(-90, 90 + res, res)
    Lon, Lat = np.meshgrid(lons, lats)
    phi, theta = np.radians(Lon), np.radians(90 - Lat)

    # Vectorized Field Calculation
    x = R_obs * np.sin(theta) * np.cos(phi)
    y = R_obs * np.sin(theta) * np.sin(phi)
    z = R_obs * np.cos(theta)
    
    R_vec = np.stack([x, y, z], axis=-1)
    M_vector = (-Q_val * B_bg_xyz).real
    
    R_mag = np.linalg.norm(R_vec, axis=-1)
    M_dot_R = np.sum(M_vector * R_vec, axis=-1)
    B_ind_xyz = ((3 * M_dot_R[..., None] * R_vec) / R_mag[..., None]**2 - M_vector) * (R_E / R_obs)**3
    B_total_mag = np.linalg.norm(B_bg_xyz + B_ind_xyz, axis=-1)

    # STABILITY CHECK: Ensure B_total_mag contains no NaNs before plotting
    if not np.all(np.isfinite(B_total_mag)):
        print(f"Warning: Non-finite values detected for Sigma={sigma_ocean}. Skipping.")
        return

    # 2. Plotting
    fig = plt.figure(figsize=(16, 7))
    
    # Panel (a)
    ax1 = fig.add_subplot(1, 2, 1)
    z_norm = np.linspace(0, 1.1, 1000)
    sig_prof = np.where((z_norm > r1/R_E) & (z_norm < r2/R_E), sigma_ocean, 0)
    ax1.plot(z_norm, sig_prof, 'k', lw=2)
    ax1.fill_between(z_norm, sig_prof, color='royalblue', alpha=0.3)
    ax1.set_title(f"(a) $\sigma$ Profile: $T_{{ocean}}$={ocean_thickness}km", loc='left', fontweight='bold')
    ax1.set_xlabel("Normalized Radius ($r/R_E$)")
    ax1.set_ylabel("Conductivity (S/m)")
    ax1.set_ylim(-0.1, sigma_ocean + 1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel (b)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
    B_diff = B_total_mag - B0
    max_dev = max(abs(B_diff.min()), abs(B_diff.max()))
    
    if max_dev < 1e-4:
        norm = mcolors.Normalize(vmin=B0 - 0.1, vmax=B0 + 0.1)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=B0 - max_dev, vcenter=B0, vmax=B0 + max_dev)

    mesh = ax2.pcolormesh(Lon, Lat, B_total_mag, transform=ccrs.PlateCarree(), 
                          cmap='bwr', norm=norm, shading='auto')
    
    ax2.set_title(f"(b) $|B|$ Magnitude ($\sigma$={sigma_ocean}, A={np.abs(Q_val):.2f})", loc='left', fontweight='bold')
    ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='grey', alpha=0.3, linestyle='--')
    
    cb = plt.colorbar(mesh, ax=ax2, orientation='horizontal', fraction=0.06, pad=0.1)
    cb.set_label('Total Field Strength (nT)')

    filename = f"Europa_S{sigma_ocean}_Ot{ocean_thickness}_It{ice_thickness}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 3. Execution Loop
# ==========================================
sigmas = [0.1, 1.0, 5.0]
ocean_thicknesses = [50, 100]
ice_thicknesses = [10, 20]
alt = 100.0

for s in sigmas:
    for ot in ocean_thicknesses:
        for it in ice_thicknesses:
            try:
                print(f"Processing: Sigma={s}, Ocean={ot}, Ice={it}...")
                save_europa_figure(s, it, ot, alt)
            except Exception as e:
                print(f"Error on Sigma={s}, Ocean={ot}, Ice={it}: {e}")

print("\nProcessing complete.")