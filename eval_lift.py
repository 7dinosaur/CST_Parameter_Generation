"""
Evaluate: iterate AoA to find CL~0.106, verify cabin + smoothness
Mach 1.8, 18km, 125t lift design
"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from aircraft_gene import Aircraft
from gene_new_para import gen_parameters
import subprocess as sp

# ============================================================
# 1. Generate 5th-order CST params
# ============================================================
print("=== Generating 5th-order CST params ===")
para = gen_parameters(5)
bwb = Aircraft(para.copy())
print(f"Params: {para.shape}")

# Planform area (half, trapezoidal)
y = para[:, 0]
chords = para[:, -4] - para[:, -5]
S_half_geom = sum(0.5*(chords[i]+chords[i+1])*(y[i+1]-y[i]) for i in range(len(y)-1))
S_full = 2 * S_half_geom
print(f"Geom: S_half={S_half_geom:.1f}m2, S_full={S_full:.1f}m2")

# ============================================================
# 2. Design conditions
# ============================================================
M = 1.8; H = 18000; rho = 0.121647; p_inf = 7565.23
a_sound = 295.07  # speed of sound at 18km
q = 0.5 * rho * (M * a_sound)**2
L_target = 125000 * 9.81
CL_target = 0.106
S_half_ref = L_target / (CL_target * q) / 2
print(f"q={q:.1f}Pa, L_target={L_target:.0f}N, S_half_ref={S_half_ref:.1f}m2")

# ============================================================
# 3. Cabin + smoothness
# ============================================================
bwb_check = Aircraft(para.copy())
bwb_check.gene_simple_mesh(81, 81, aoa=0.0)
cabin_ok = bwb_check.search_cabin()
lap = bwb_check.Laplace()
smooth = bwb_check.if_smooth()
print(f"Cabin: {'PASS' if cabin_ok else 'FAIL'}")
print(f"Laplace: {lap[0]:.4f} / {lap[1]:.4f}")
print(f"Smoothness: [{smooth[0]:.4f}, {smooth[1]:.4f}]")

# Thickness check
print("Stations:")
for row in bwb_check.origin_para:
    y_st = row[0]; le = row[-5]; te = row[-4]; chord = te - le
    u, l = bwb_check.cst_rec(row, bwb_check.N1, bwb_check.N2, n_points=100)
    max_t = np.max(u[1,:]-l[1,:])
    cm = (u[0]>=22)&(u[0]<=60.8)
    u_min = np.min(u[1,cm]) if cm.any() else np.nan
    l_max = np.max(l[1,cm]) if cm.any() else np.nan
    print(f"  y={y_st:.0f}: chord={chord:.1f}, t/c={max_t/chord*100:.2f}%, u_cab={u_min:.2f}, l_cab={l_max:.2f}")

# ============================================================
# 4. AoA sweep with FABOOM
# ============================================================
# Update FABoom.in (AoA=0, mesh rotation handles it)
faboom_in = f"""Case name
Tu144
2Ma
====Mach Number====AOA (deg)====Height (m)====Density (kg/m3)====Pressure (Pa)====ROVERL====PHI (deg)====
        {M:.1f}             0.00        {H}         {rho}           {p_inf}          3          0.
====Aircraft Length (m)====Model Length(m)====aircraft height(m)===half reference area(m2)
          72           72                     4             {S_half_ref:.1f}
====cut section number=====
                0
====cut y position=======
3.0  6.0   9.0
====Area_Num_Interpolation====NX_Nearfield====
                     40                              500
====volume or lift or area=====predict or design====
2            1"""
with open(r"FABOOM_test\indata\FABoom.in", 'w') as f:
    f.write(faboom_in)
print(f"\nFABoom.in updated: S_half_ref={S_half_ref:.1f}m2")

bwb_panel = Aircraft(para.copy())

aoa_list = [2.0, 2.5, 3.0, 3.5, 4.0]
results = []

for aoa in aoa_list:
    print(f"\n--- AoA = {aoa:.1f} deg ---")
    try:
        # Write panel mesh with AoA baked into rotation
        bwb_panel.write_mesh("panel", r"FABOOM_test\indata\geo.x", aoa=aoa)

        # Run FABOOM
        proc = sp.run([r"FABOOM_test\FABOOM.exe"], cwd=r"FABOOM_test",
                       capture_output=True, text=True, timeout=300)

        # Read CL
        lift_data = np.loadtxt(r"FABOOM_test\A502\Lift distribution.dat")
        CL = lift_data[-1, 1]
        err = CL - CL_target
        print(f"CL={CL:.6f}, target={CL_target}, error={err:+.4f}")
        results.append((aoa, CL, err))
    except Exception as e:
        print(f"FAILED: {e}")
        results.append((aoa, None, None))

# ============================================================
# 5. Summary
# ============================================================
print("\n=== Summary ===")
print(f"{'AoA':>6s} {'CL':>10s} {'Error':>10s}")
for aoa, cl, err in results:
    if cl is not None:
        print(f"{aoa:6.2f} {cl:10.6f} {err:+10.6f}")
    else:
        print(f"{aoa:6.2f} {'FAIL':>10s}")

valid = [(a,c,e) for a,c,e in results if c is not None]
if valid:
    best = min(valid, key=lambda x: abs(x[2]))
    print(f"\nBest AoA: {best[0]:.2f} deg, CL={best[1]:.6f}, err={best[2]:+.4f}")
