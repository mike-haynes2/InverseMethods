import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import spherical_jn, spherical_yn
import cartopy.crs as ccrs
import os

# ==========================================
# 1. Physics Engine (Stabilized)
# ==========================================
from scipy.special import jv

def calculate_Q_zimmer_fixed(f, sigma, r1_km, r2_km, R_target_km):
    # """
    # Stabilized Zimmer (2000) induction using jv for exact fractional orders.
    # f: frequency (Hz)
    # sigma: conductivity (S/m)
    # r1_km, r2_km: inner/outer ocean radii (km)
    # R_target_km: observer altitude (km)
    # """
    # mu0 = 4 * np.pi * 1e-7
    # omega = 2 * np.pi * f
    # # Complex wave number k from Zimmer image
    # k = (1 - 1j) * np.sqrt(mu0 * sigma * omega / 2.0)
    
    # r0 = r2_km * 1000.0  # Outer ocean
    # r1 = r1_km * 1000.0  # Inner ocean
    # rm = R_target_km * 1000.0 # Observer
    
    # z0, z1 = r0 * k, r1 * k
    
    # # Zimmer Eq (6): Reflection coefficient R at core boundary
    # # R = (r1*k*J_-5/2) / (3*J_3/2 - r1*k*J_1/2)
    # R_num = z1 * jv(-2.5, z1)
    # R_den = 3 * jv(1.5, z1) - z1 * jv(0.5, z1)
    # R = R_num / R_den
    
    # # Zimmer Eq (5): Complex response Ae^(i*phi) at observer radius rm
    # # Note: Includes (r0/rm)^3 distance fall-off
    # ae_num = R * jv(2.5, z0) - jv(-2.5, z0)
    # ae_den = R * jv(0.5, z0) - jv(-0.5, z0)
    
    # Q_complex = (r0 / rm)**3 * (ae_num / ae_den)
    
    # # Check for numerical instability: A cannot exceed (r0/rm)^3
    # limit = (r0 / rm)**3
    # if np.abs(Q_complex) > limit:
    #     return limit + 0j # Clamp to physical maximum
        
    # return Q_complex
    if sigma < 1e-6: return 0.0 + 0.0j # Physical limit: no induction for insulators
    
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f
    k = (1 - 1j) * np.sqrt(mu0 * sigma * omega / 2.0)
    
    r0, r1, rm = r2_km * 1000, r1_km * 1000, R_target_km * 1000
    z0, z1 = r0 * k, r1 * k
    
    # Zimmer Eq (6): R coefficient
    R_num = z1 * jv(-2.5, z1)
    R_den = 3 * jv(1.5, z1) - z1 * jv(0.5, z1)
    R = R_num / (R_den + 1e-20) # Guard against div by zero
    
    # Zimmer Eq (5): Complex Response
    ae_num = R * jv(2.5, z0) - jv(-2.5, z0)
    ae_den = R * jv(0.5, z0) - jv(-0.5, z0)
    
    Q_complex = (r0 / rm)**3 * (ae_num / (ae_den + 1e-20))
    
    return Q_complex if np.isfinite(Q_complex) else 0.0j

# ==========================================
# 2. Map Generation Function
# ==========================================
def save_europa_figure(sigma_ocean, ice_thickness, ocean_thickness, alt_km, output_dir='figs'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- Constants and Geometry ---
    R_E, B0 = 1560.0, 210.0
    r2, r1 = R_E - ice_thickness, R_E - ice_thickness - ocean_thickness
    R_obs = R_E + alt_km
    freq = 1 / (11.1 * 3600)

    # --- Physics Engine (Zimmer Eqs 5 & 6) ---
    # Implement the calculation exactly as defined in Zimmer et al. 2000
    Q_val = calculate_Q_zimmer_fixed(freq, sigma_ocean, r1, r2, R_obs)
    A_mag = np.abs(Q_val)
    
    # --- Spatial Field Calculation ---
    res = 5 # Higher resolution for final figures
    lons = np.arange(-180, 180 + res, res)
    lats = np.arange(-90, 90 + res, res)
    Lon, Lat = np.meshgrid(lons, lats)
    phi, theta = np.radians(Lon), np.radians(90 - Lat)
    
    R_vec = R_obs * np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
    # The complex Q determines the magnitude and phase of the induced moment M
    M_vector = (-Q_val * np.array([0, B0, 0])).real
    R_mag = np.linalg.norm(R_vec, axis=-1)
    M_dot_R = np.sum(M_vector * R_vec, axis=-1)
    B_ind = ((3 * M_dot_R[..., None] * R_vec) / R_mag[..., None]**2 - M_vector) * (R_E / R_obs)**3
    B_total_mag = np.linalg.norm(np.array([0, B0, 0]) + B_ind, axis=-1)

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 8))
    
    # Plot (a): Conductivity Profile
    ax1 = fig.add_subplot(1, 2, 1)
    z_norm = np.linspace(0, 1.2, 500)
    sig_prof = np.where((z_norm > r1/R_E) & (z_norm < r2/R_E), sigma_ocean, 0)
    ax1.plot(z_norm, sig_prof, 'k', lw=2)
    ax1.fill_between(z_norm, sig_prof, color='royalblue', alpha=0.3)
    
    # Requested formatting
    ax1.set_ylim(0, 1.5) # Fixed conductivity range
    ax1.set_xlim(0.8, 1.1) # Zoomed into the shell region
    ax1.set_xlabel("Normalized Radius ($r/R_E$)", fontsize=12)
    ax1.set_ylabel("Conductivity $\sigma$ (S/m)", fontsize=12)
    ax1.set_title(f"(a) Profile: $\sigma$={sigma_ocean} S/m", loc='left', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot (b): Mollweide Projection
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
    
    # Fixed field range: B0 +/- 200 nT (10 to 410 nT)
    # This prevents the monotonic error and keeps colors consistent
    norm = mcolors.TwoSlopeNorm(vmin=40, vcenter=210, vmax=380)
    
    mesh = ax2.pcolormesh(Lon, Lat, B_total_mag, transform=ccrs.PlateCarree(), 
                          cmap='bwr', norm=norm, shading='auto')
    
    # Gridlines: gray, semi-transparent, dashed
    gl = ax2.gridlines(draw_labels=False, color='gray', alpha=0.5, linestyle='--')
    
    # Colorbar and Labels
    cbar = plt.colorbar(mesh, ax=ax2, orientation='horizontal', pad=0.08, extend='both')
    cbar.set_label("Total Magnetic Field Magnitude $|B|$ (nT)", fontsize=12)
    
    ax2.set_title(f"(b) Surface Field Projection (A={A_mag:.3f})", loc='left', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, f"Europa_S{sigma_ocean}_Ot{ocean_thickness}.png"), 
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return A_mag
# def save_europa_figure(sigma_ocean, ice_thickness, ocean_thickness, alt_km, output_dir='figs'):
#     if not os.path.exists(output_dir): os.makedirs(output_dir)

#     R_E, B0 = 1560.0, 210.0
#     r2, r1 = R_E - ice_thickness, R_E - ice_thickness - ocean_thickness
#     R_obs = R_E + alt_km
#     freq = 1 / (11.1 * 3600)

#     Q_val = calculate_Q_zimmer_fixed(freq, sigma_ocean, r1, r2, R_obs)
#     A_mag = np.abs(Q_val)
    
#     res = 10 
#     lons = np.arange(-180, 180 + res, res)
#     lats = np.arange(-90, 90 + res, res)
#     Lon, Lat = np.meshgrid(lons, lats)
#     phi, theta = np.radians(Lon), np.radians(90 - Lat)
    
#     R_vec = R_obs * np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
#     M_vector = (-Q_val * np.array([0, B0, 0])).real
#     R_mag = np.linalg.norm(R_vec, axis=-1)
#     M_dot_R = np.sum(M_vector * R_vec, axis=-1)
#     B_ind = ((3 * M_dot_R[..., None] * R_vec) / R_mag[..., None]**2 - M_vector) * (R_E / R_obs)**3
#     B_total_mag = np.linalg.norm(np.array([0, B0, 0]) + B_ind, axis=-1)

#     fig = plt.figure(figsize=(16, 7))
#     ax1 = fig.add_subplot(1, 2, 1)
#     z_norm = np.linspace(0, 1.1, 500)
#     sig_prof = np.where((z_norm > r1/R_E) & (z_norm < r2/R_E), sigma_ocean, 0)
#     ax1.plot(z_norm, sig_prof, 'k', lw=2); ax1.fill_between(z_norm, sig_prof, color='royalblue', alpha=0.3)
#     ax1.set_title(f"(a) Profile: $\sigma$={sigma_ocean}", loc='left', fontweight='bold')
    
#     ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
#     max_dev = max(abs(B_total_mag.min()-B0), abs(B_total_mag.max()-B0))
#     norm = mcolors.TwoSlopeNorm(vmin=B0-max_dev-0.1, vcenter=B0, vmax=B0+max_dev+0.1)
#     mesh = ax2.pcolormesh(Lon, Lat, B_total_mag, transform=ccrs.PlateCarree(), cmap='bwr', norm=norm, shading='auto')
#     ax2.set_title(f"(b) $|B|$ (A={A_mag:.3f})", loc='left', fontweight='bold')
    
#     plt.savefig(os.path.join(output_dir, f"Europa_S{sigma_ocean}_Ot{ocean_thickness}_It{ice_thickness}.png"), bbox_inches='tight')
#     plt.close(fig)
    
#     return A_mag

# ==========================================
# 3. Execution Loop
# ==========================================
sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] # Log-spaced for line plot
ocean_thicknesses = [10, 50, 100, 200]
ice_fixed = 20.0
alt = 100.0

results = {ot: [] for ot in ocean_thicknesses}

print("Simulating parameter space...")
for ot in ocean_thicknesses:
    for s in sigmas:
        A_val = save_europa_figure(s, ice_fixed, ot, alt)
        results[ot].append(A_val)

# ==========================================
# 4. Summary Plots
# ==========================================
fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap
results_matrix = np.array([results[ot] for ot in ocean_thicknesses]).T # Transpose for Sigma on Y
im = ax_heat.imshow(results_matrix, aspect='auto', origin='lower', cmap='plasma', vmin=0., vmax=1.)
ax_heat.set_yticks(range(len(sigmas)))
ax_heat.set_yticklabels(sigmas)
ax_heat.set_xticks(range(len(ocean_thicknesses)))
ax_heat.set_xticklabels(ocean_thicknesses)
ax_heat.set_title("2D Induction Amplitude (A) Heatmap", fontweight='bold')
ax_heat.set_ylabel("Conductivity $\sigma$ (S/m)")
ax_heat.set_xlabel("Ocean Thickness (km)")
plt.colorbar(im, ax=ax_heat, label='Amplitude Ratio (A)')

# Line Plot
for ot in ocean_thicknesses:
    ax_line.plot(sigmas, results[ot], '-o', label=f'Ocean = {ot} km', lw=2)

ax_line.set_xscale('log')
ax_line.set_title("1D Induction Sensitivity Curves", fontweight='bold')
ax_line.set_xlabel("Ocean Conductivity $\sigma$ (S/m) [Log Scale]")
ax_line.set_ylabel("Induction Amplitude (A)")
ax_line.set_ylim(0, 1.05)
ax_line.grid(True, which="both", ls="-", alpha=0.2)
ax_line.legend()

plt.tight_layout()
plt.savefig('figs/Summary_Analysis.png', dpi=200)
plt.show()

print("All figures and summary analysis saved in 'figs/'.")