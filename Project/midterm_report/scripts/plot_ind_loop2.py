import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import spherical_jn, spherical_yn
import cartopy.crs as ccrs
import os

# ==========================================
# 1. Physics Engine: Zimmer et al. (2000)
# ==========================================
def calculate_Q_complex(f, sigma, r1_km, r2_km, R_target_km):
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f
    k = np.sqrt(-1j * mu0 * sigma * omega)
    
    a, b = r2_km * 1000, r1_km * 1000
    ka, kb = k * a, k * b
    
    j1_ka, y1_ka = spherical_jn(1, ka), spherical_yn(1, ka)
    j1_kb, y1_kb = spherical_jn(1, kb), spherical_yn(1, kb)
    
    dj_dz_ka = spherical_jn(1, ka) + ka * spherical_jn(1, ka, derivative=True)
    dy_dz_ka = spherical_yn(1, ka) + ka * spherical_yn(1, ka, derivative=True)
    dj_dz_kb = spherical_jn(1, kb) + kb * spherical_jn(1, kb, derivative=True)
    dy_dz_kb = spherical_yn(1, kb) + kb * spherical_yn(1, kb, derivative=True)

    alpha = dj_dz_kb / dy_dz_kb
    
    Q_surface = ((ka * j1_ka - dj_dz_ka) - alpha * (ka * y1_ka - dy_dz_ka)) / \
                ((ka * j1_ka + dj_dz_ka) - alpha * (ka * y1_ka + dy_dz_ka))

    return Q_surface * (a / (R_target_km * 1000))**3

# ==========================================
# 2. Plotting Function
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

    R_vec = R_obs * np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
    M_vector = (-Q_val * B_bg_xyz).real
    
    R_mag = np.linalg.norm(R_vec, axis=-1)
    M_dot_R = np.sum(M_vector * R_vec, axis=-1)
    B_ind_xyz = ((3 * M_dot_R[..., None] * R_vec) / R_mag[..., None]**2 - M_vector) * (R_E / R_obs)**3
    B_total_mag = np.linalg.norm(B_bg_xyz + B_ind_xyz, axis=-1)

    # 2. Plotting
    fig = plt.figure(figsize=(16, 7))
    
    # Panel (a)
    ax1 = fig.add_subplot(1, 2, 1)
    z_norm = np.linspace(0, 1.1, 1000)
    sig_prof = np.where((z_norm > r1/R_E) & (z_norm < r2/R_E), sigma_ocean, 0)
    ax1.plot(z_norm, sig_prof, 'k', lw=2)
    ax1.fill_between(z_norm, sig_prof, color='royalblue', alpha=0.3)
    ax1.set_title(f"(a) $\sigma$ Profile: $T_{{ocean}}$={ocean_thickness}km", fontsize=14)
    ax1.set_xlim(0, 1.1); ax1.set_ylim(-0.1, sigma_ocean + 1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel (b)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
    B_diff = B_total_mag - B0
    max_dev = max(abs(B_diff.min()), abs(B_diff.max()))
    
    # STABILITY: Guard against collapsed normalization range
    if max_dev < 1e-6:
        norm = mcolors.Normalize(vmin=B0 - 1, vmax=B0 + 1)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=B0 - max_dev, vcenter=B0, vmax=B0 + max_dev)

    mesh = ax2.pcolormesh(Lon, Lat, B_total_mag, transform=ccrs.PlateCarree(), cmap='bwr', norm=norm, shading='auto')
    ax2.set_title(f"(b) $|B|$ Magnitude ($\sigma$={sigma_ocean}, Alt={alt_km}km)", fontsize=14)
    ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='grey', alpha=0.3, linestyle='--')
    
    cb = plt.colorbar(mesh, ax=ax2, orientation='horizontal', fraction=0.06, pad=0.1)
    cb.set_label('Total Field Strength (nT)')

    # Save and Close
    filename = f"Europa_S{sigma_ocean}_Ot{ocean_thickness}_It{ice_thickness}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig) # CRITICAL: Memory management

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

print("\nProcessing complete. Check the 'figs/' folder.")