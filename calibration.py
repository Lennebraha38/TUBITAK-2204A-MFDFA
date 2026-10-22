"""
MF-DFA KALIBRASYON CETVELI
==========================
Bilinen dogasi olan surecler icin Delta_alpha olcup bir "cetvel" olusturur.
Amac: gercek veride gorulen Delta_alpha'nin hangi rejime dustugunu
gostermek ve motorun matematik dogrulugunu kanitlamak.

Referans surecler
-----------------
  - Beyaz gurultu / tekduze : tamamen rastgele          -> Delta_alpha ~ 0
  - fGn(H)                  : tekli fraktal, h(2)=H     -> Delta_alpha kucuk
  - AR(1)                   : kisa erimli dogrusal bellek
  - p-model (kaskad)        : DETERMINISTIK coklu fraktal -> Delta_alpha ~ 1.2
  - MRW(sigma)              : STOKASTIK coklu fraktal     -> Delta_alpha buyuk

Uretir: results/calibration_table.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import (mfdfa, spectrum_width, pmodel_cascade, theoretical_alpha,
                        surrogate_test, fgn, ar1, white_noise, mrw)

N = 8192
Q = np.linspace(-5, 5, 21)
I2 = int(np.where(np.isclose(Q, 2.0))[0][0])
rng = np.random.default_rng(0)

rows = []


def report(name, x, group, do_surr=False, note=""):
    r = mfdfa(x, Q, order=2, n_scales=30)
    da = spectrum_width(r)
    h2 = r["h_q"][I2]
    z = np.nan
    if do_surr:
        st = surrogate_test(x, Q, n_surr=50, kind="iaaft", seed=1)
        z = st["z"]
    rows.append(dict(surec=name, grup=group, N=len(x), delta_alpha=round(da, 4),
                     h_q2=round(h2, 4), iaaft_z=round(z, 2) if np.isfinite(z) else "",
                     not_=note))
    print(f"{name:26s} {group:22s} N={len(x):5d}  DA={da:7.4f}  h(q=2)={h2:6.3f}"
          + (f"  IAAFT z={z:+6.2f}" if np.isfinite(z) else ""), flush=True)


print("=" * 92)
print("MF-DFA KALIBRASYON CETVELI  (N=8192, q in [-5,5], s_min=32, 30 olcek)")
print("=" * 92)
print(f"{'Surec':26s} {'Grup':22s} {'N':>6s}  {'Delta_a':>7s}  {'h(q=2)':>6s}")
print("-" * 92)

# --- 1. Tamamen rastgele ---
report("Beyaz gurultu", white_noise(N, rng), "rastgele", do_surr=True)
report("Tekduze dagilim", rng.uniform(0, 1, N), "rastgele")
report("Rastgele yuruyus", np.cumsum(rng.standard_normal(N)), "rastgele")

# --- 2. Tekli fraktal: Hurst kurtarimi ---
for H in (0.3, 0.5, 0.7, 0.9):
    report(f"fGn H={H}", fgn(N, H, rng), "tekli fraktal",
           note=f"beklenen h(2)={H}")

# --- 3. Dogrusal bellek ---
for phi in (0.5, 0.9, 0.98):
    report(f"AR(1) phi={phi}", ar1(N, phi, rng), "dogrusal bellek")

# --- 4. Deterministik coklu fraktal ---
# p=0.5 kaskadi sabit bir seri uretir (tum degerler esit) -> atlanir.
for p in (0.4, 0.3, 0.2):
    x = pmodel_cascade(N, p)
    theo = float(theoretical_alpha(Q, p).max() - theoretical_alpha(Q, p).min())
    report(f"p-model p={p}", x, "coklu fraktal (det.)",
           do_surr=(p == 0.3), note=f"teorik DA={theo:.4f}")

# --- 5. Stokastik coklu fraktal ---
for sg in (0.3, 0.5, 0.8):
    report(f"MRW sigma={sg}", mrw(N, sigma=sg, rng=rng),
           "coklu fraktal (stok.)", do_surr=(sg == 0.8))

df = pd.DataFrame(rows)
out = r"C:\TUBITAK_Proje\results\calibration_table.csv"
df.to_csv(out, index=False)

print("-" * 92)
print("YORUM")
print("  Delta_alpha ~ 0.00-0.20  -> tekli fraktal / rastgele  (YAPI YOK)")
print("  Delta_alpha ~ 0.20-0.50  -> zayif-orta coklu fraktal")
print("  Delta_alpha ~ 1.0-1.3    -> guclu coklu fraktal (p-model, MRW)")
print()
print("  ONEMLI: p-model'in genlik dagilimi AYRIK oldugu icin IAAFT surrogu")
print("  onun Delta_alpha'sini taklit edebilir (z~0.6). Bu yuzden IAAFT tek")
print("  basina yeterli degildir; shuffle ve faz-rastgele null'lari da")
print("  kullanilmalidir. Ayrintili test: tests/test_mfdfa.py")
print("-" * 92)
print("Kaydedildi:", out)
