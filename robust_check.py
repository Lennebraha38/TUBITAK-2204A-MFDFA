"""
Robustluk: buyuk Delta_alpha global trendden mi geliyor, yoksa gercek mi?
- Ham seri vs polinom detrend edilmis seri
- Artim (increment) serisi
- Faz ortalamalari
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, surrogate_test

s = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                index_col=0, parse_dates=True)
Q = np.linspace(-5, 5, 21)
x = s["distress"].values.astype(float)
t = np.arange(len(x))
N = len(x)

print(f"N={N}")
print(f"distress: ilk 24s ort={x[:1440].mean():.4f} | "
      f"son 48s ort={x[-2880:].mean():.4f} | genel={x.mean():.4f}")
print(f"distress: min={x.min():.4f} max={x.max():.4f} std={x.std():.4f}")
print()


def detrend(y, deg):
    c = np.polyfit(t, y, deg)
    return y - np.polyval(c, t)


cases = [
    ("ham (raw)", x),
    ("lineer detrend", detrend(x, 1)),
    ("kuadratik detrend", detrend(x, 2)),
    ("kubik detrend", detrend(x, 3)),
    ("1. fark (diff)", np.diff(x)),
    ("gunluk ortalama cikar", x - pd.Series(x, index=s.index).resample("1D").transform("mean").values),
]

print(f"{'Varyant':24s} {'DA':>8s} {'Dh':>8s} {'h(q=2)':>8s} {'IAAFT z':>9s} {'p':>8s}")
print("-" * 74)
i2 = int(np.where(np.isclose(Q, 2.0))[0][0])
for name, y in cases:
    y = np.asarray(y, dtype=float)
    r = mfdfa(y, Q)
    st = surrogate_test(y, Q, n_surr=50, kind="iaaft", seed=1)
    print(f"{name:24s} {spectrum_width(r):8.4f} "
          f"{r['h_q'][0]-r['h_q'][-1]:8.4f} {r['h_q'][i2]:8.4f} "
          f"{st['z']:+9.2f} {st['p_value']:8.4f}")
