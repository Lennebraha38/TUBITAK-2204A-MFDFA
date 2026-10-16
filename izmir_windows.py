"""
Izmir penceresi testi: seyrek olay-oncesi donem mi Delta_alpha'yi sisiriyor?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, surrogate_test

Q = np.linspace(-5, 5, 21)
EV = pd.Timestamp("2020-10-30T11:51", tz="UTC")

s = pd.read_csv(r"C:\TUBITAK_Proje\results\izmir2020_series_1min.csv",
                index_col=0, parse_dates=True)

windows = [
    ("ham TUM (tam seri)", None, None),
    ("[olay-1s, son]", ("2020-10-30T10:51", None)),
    ("[olay, son]", ("2020-10-30T11:51", None)),
    ("[olay+12s, son]", ("2020-10-30T23:51", None)),
    ("[olay, +24s] AKUT", ("2020-10-30T11:51", "2020-10-31T11:51")),
    ("[olay, +72h]", ("2020-10-30T11:51", "2020-11-02T11:51")),
    ("ilk 48s (spin-up sonrasi)", ("2020-10-31T12:00", "2020-11-02T12:00")),
]

print(f"{'pencere':32s} {'N':>6s} {'bos%':>6s} {'DA':>8s} {'h(q2)':>7s} {'IAAFT z':>9s}")
print("-" * 74)
i2 = int(np.where(np.isclose(np.linspace(-5, 5, 21), 2.0))[0][0])
for lbl, (a, b) in [(l, (None, None) if w is None else w)
                    for l, w in [(x[0], x[1]) for x in windows]]:
    src = pd.read_csv(r"C:\TUBITAK_Proje\results\izmir2020_series_1min.csv",
                      index_col=0, parse_dates=True)
    if a:
        src = src[src.index >= pd.Timestamp(a, tz="UTC")]
    if b:
        src = src[src.index <= pd.Timestamp(b, tz="UTC")]
    cnt = src["count"].values
    x = src["distress"].values.astype(float)
    nempty = float((cnt == 0).mean() * 100)
    if len(x) < 128:
        print(f"{lbl:32s} {len(x):6d}  (cok kisa)")
        continue
    r = mfdfa(x, Q)
    st = surrogate_test(x, Q, n_surr=50, kind="iaaft", seed=1)
    print(f"{lbl:32s} {len(x):6d} {nempty:6.1f} {spectrum_width(r):8.4f} "
          f"{r['h_q'][i2]:7.3f} {st['z']:+9.2f}")
