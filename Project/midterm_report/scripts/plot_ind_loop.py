import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import spherical_jn, spherical_yn
import cartopy.crs as ccrs
import os

# ==========================================
# 1. Physics Engine: Zimmer et al. (2000)
# ==========================================
def calculate_Q_zimmer(f, sigma, r1_km, r2_km):
    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f
    k = np.sqrt(1j * mu0 * sigma * omega)
    
    a, b = r2_km * 1000, r1_km * 1000
    ka, kb = k * a, k * b
    
    def d_zjn(n, z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def d_zyn(n, z):
        return spherical_yn(n, z) + z * spherical_yn(n, z, derivative=True)

    alpha = d_zjn(1, kb) / d_zyn(1, kb)
    
    num = (ka * spherical_jn(1, ka) - d_zjn(1, ka)) - alpha * (ka * spherical_yn(1, ka) - d_zyn(1, ka))
    den = (ka * spherical_jn(1, ka) + d_zjn(1, ka)) - alpha * (ka * spherical_yn(1, ka) + d_zyn(1, ka))
    
    return num / den

# ==========================================
# 2. Main Plotting Function
# ==========================================
def generate_europa_induction_plot(sigma_ocean, ice_thick, ocean_thick, output_folder='figs'):
    # Setup Radii
    R_E = 1560.0
    r2 = R_E - ice_thick
    r1 = r2 - ocean_thick
    B0_mag = 210.0
    freq = 1 / (11.1 * 3600)

    # Calculate Induction Parameters
    Q = calculate_Q_zimmer(freq, sigma_ocean, r1, r2)
    A, phi = np.abs(Q), np.angle(Q)

    # Spatial Grid
    lon = np.linspace(-180, 180, 180)
    lat = np.linspace(-90, 90, 90)
    Lon, Lat = np.meshgrid(lon, lat)
    phi_grid = np.radians(Lon)
    theta_grid = np.radians(90 - Lat)

    # Field Magnitude calculation (assuming B_bg in Z-direction for visualization)
    B_total = B0_mag * np.sqrt((1 - A*np.cos(phi_grid))**2 * np.cos(theta_grid)**2 + 
                               (1 + 0.5*A*np.cos(phi_grid))**2 * np.sin(theta_grid)**2)

    # Figure Setup
    fig = plt.figure(figsize=(14, 6))
    
    # --- Panel (a): Conductivity Profile ---
    ax1 = fig.add_subplot(1, 2, 1)
    r_norm = np.linspace(0, 1, 500)
    sig_prof = np.zeros_like(r_norm)
    sig_prof[(r_norm >= r1/R_E) & (r_norm <= r2/R_E)] = sigma_ocean
    ax1.plot(r_norm, sig_prof, color='blue', lw=2)
    ax1.fill_between(r_norm, sig_prof, color='skyblue', alpha=0.3)
    ax1.set_title(f"(a) Profile: Ocean={ocean_thick}km, $\sigma$={sigma_ocean}", loc='left', fontweight='bold')
    ax1.set_xlabel("Normalized Radius ($r/R_E$)")
    ax1.set_ylabel("Conductivity (S/m)")
    ax1.set_ylim(-0.1, max(5, sigma_ocean + 1))
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Panel (b): Mollweide Map ---
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
    v_min, v_max = B_total.min(), B_total.max()
    
    # Fix for TwoSlopeNorm error if induction is negligible
    if abs(v_max - v_min) < 1e-2:
        norm = mcolors.Normalize(vmin=B0_mag - 1, vmax=B0_mag + 1)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=B0_mag, vmax=v_max)

    mesh = ax2.pcolormesh(Lon, Lat, B_total, transform=ccrs.PlateCarree(),
                          cmap='bwr', norm=norm, shading='auto')
    
    ax2.set_global()
    ax2.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax2.set_title(f"(b) Surface $|B|$ (A={A:.2f}, $\phi$={np.degrees(phi):.1f}°)", loc='left', fontweight='bold')

    cb = plt.colorbar(mesh, ax=ax2, orientation='horizontal', pad=0.08, aspect=30)
    cb.set_label("Total Field Strength (nT)")

    # Save logic
    filename = f"Europa_S{sigma_ocean}_T{ocean_thick}_I{ice_thick}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close to save memory during loop
    print(f"Saved: {filepath}")

# ==========================================
# 3. Execution Loop
# ==========================================
if __name__ == "__main__":
    # Create directory if it doesn't exist
    if not os.path.exists('figs'):
        os.makedirs('figs')

    # Define your parameter grid
    conductivities = [0.1, 0.5, 1.0, 2., 3., 4., 5.0]     # S/m
    ocean_thicknesses = [20, 50, 100, 150]  # km
    ice_thicknesses = [10, 30, 50]          # km

    for sig in conductivities:
        for thick in ocean_thicknesses:
            for ice in ice_thicknesses:
                generate_europa_induction_plot(sig, ice, thick)

    print("\nAll figures generated in /figs folder.")