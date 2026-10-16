"""
Delta_alpha duyarlilik analizi: q araligi ve olcek araligina gore.
Boyuk Delta_alpha bir artefakt mi, yoksa saglam mi?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, h_spread, pmodel_cascade

s = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                index_col=0, parse_dates=True)
x = s["distress"].values.astype(float)
t = np.arange(len(x))
xq = x - np.polyval(np.polyfit(t, x, 2), t)  # kuadratik detrend

ref = pmodel_cascade(8192, 0.3)  # bilinen multifraktal referans

print("=== q ARALIGI DUYARLILIGI (n_scales=30, s_max_frac=4) ===")
print(f"{'q araligi':>12s} {'ham DA':>9s} {'ham Dh':>9s} {'kuad-detr DA':>13s} "
      f"{'p-model DA':>11s}")
print("-" * 62)
for qmax in (2, 3, 4, 5, 8, 10):
    Q = np.linspace(-qmax, qmax, 2 * qmax + 1)
    rh = mfdfa(x, Q)
    rq = mfdfa(xq, Q)
    rp = mfdfa(ref, Q)
    print(f"[-{qmax},{qmax}]".rjust(12) + f" {spectrum_width(rh):9.4f} "
          f"{h_spread(rh):9.4f} {spectrum_width(rq):13.4f} {spectrum_width(rp):11.4f}")

print()
print("=== OLCEK ARALIGI / COZUNURLUK DUYARLILIGI (q in [-5,5]) ===")
Q = np.linspace(-5, 5, 21)
print(f"{'n_scales':>9s} {'s_max_frac':>11s} {'s_min':>6s} {'ham DA':>9s} {'kuad DA':>9s}")
print("-" * 52)
for ns in (20, 30, 40, 50):
    for smf in (4, 8):
        rh = mfdfa(x, Q, n_scales=ns, s_max_frac=smf)
        rq = mfdfa(xq, Q, n_scales=ns, s_max_frac=smf)
        print(f"{ns:9d} {smf:11d} {16:6d} {spectrum_width(rh):9.4f} "
              f"{spectrum_width(rq):9.4f}")

print()
print("=== h(q) egrisi (ham, q in [-5,5]) - monoton mu? ===")
Q = np.linspace(-5, 5, 21)
rh = mfdfa(x, Q)
print("q   :", " ".join(f"{q:5.1f}" for q in Q))
print("h(q):", " ".join(f"{v:5.2f}" for v in rh["h_q"]))
