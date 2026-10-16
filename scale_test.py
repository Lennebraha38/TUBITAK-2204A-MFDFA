"""Olcek araligi robustlugu: s_min ve s_max_frac icin Delta_alpha (ham + detrend + p-model)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, pmodel_cascade, theoretical_alpha

s = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                index_col=0, parse_dates=True)
x = s["distress"].values.astype(float)
t = np.arange(len(x))
xq = x - np.polyval(np.polyfit(t, x, 2), t)
ref = pmodel_cascade(8192, 0.3)
Q = np.linspace(-5, 5, 21)
theo = theoretical_alpha(Q, 0.3)
print(f"p-model teorik DA = {theo.max()-theo.min():.4f}")
print(f"{'s_min':>6s} {'s_max_frac':>10s} {'s_max':>6s} {'ham':>8s} "
      f"{'kuad':>8s} {'p-model':>9s}")
print("-" * 56)
for smin in (16, 32, 64):
    for smf in (4, 8, 16, 32):
        rh = mfdfa(x, Q, s_min=smin, s_max_frac=smf)
        rq = mfdfa(xq, Q, s_min=smin, s_max_frac=smf)
        rp = mfdfa(ref, Q, s_min=smin, s_max_frac=smf)
        print(f"{smin:6d} {smf:10d} {8192//smf:6d} {spectrum_width(rh):8.4f} "
              f"{spectrum_width(rq):8.4f} {spectrum_width(rp):9.4f}")
