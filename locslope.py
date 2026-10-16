"""
Yerel egim analizi: log F(q,s) vs log s — olcekleme (scaling) bolgesini bul.
q degerleri icin yerel egim ds/dlog s nerede duz (plateau)?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, pmodel_cascade

s = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                index_col=0, parse_dates=True)
x = s["distress"].values.astype(float)


def Fq_curve(x, q_vals, order=2, n_scales=60, s_max_frac=4):
    x = np.asarray(x, dtype=float)
    N = len(x)
    Y = np.cumsum(x - np.mean(x))
    s_max = N // s_max_frac
    scales = np.unique(np.round(np.logspace(np.log10(8), np.log10(s_max),
                                            n_scales)).astype(int))
    Fq = np.zeros((len(q_vals), len(scales)))
    for j, ss in enumerate(scales):
        step = max(1, ss // 2)
        starts = np.arange(0, N - ss + 1, step)
        t = np.arange(ss)
        F2 = []
        for st in starts:
            seg = Y[st:st + ss]
            c = np.polyfit(t, seg, order)
            F2.append(np.mean((seg - np.polyval(c, t)) ** 2))
        F2 = np.array([v for v in F2 if v > 0])
        for i, q in enumerate(q_vals):
            if abs(q) < 1e-10:
                Fq[i, j] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                Fq[i, j] = np.mean(F2 ** (q / 2.0)) ** (1.0 / q)
    return scales, Fq


Q = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
scales, Fq = Fq_curve(x, Q)
log_s = np.log(scales)
print("Olcek listesi:", scales[:12], "...", scales[-5:])
print()
print(f"{'q':>6s} | yerel egim  d logF / d log s  (pencere=3)")
for i, q in enumerate(Q):
    lf = np.log(Fq[i])
    # yerel egim (egri fit 2. derece yerel)
    sl = np.gradient(lf, log_s)
    # yumusatma
    k = 3
    sl_s = np.convolve(sl, np.ones(k) / k, mode="same")
    print(f"\nq={q:+.0f}:")
    for j in range(0, len(scales), 4):
        print(f"   s={scales[j]:6d}  h_yerel={sl_s[j]:+.3f}")

# Kimin icin referans: p-model
print("\n\n=== REFERANS: p-model p=0.3 (temiz multifraktal) ===")
ref = pmodel_cascade(8192, 0.3)
scales2, Fq2 = Fq_curve(ref, Q)
log_s2 = np.log(scales2)
for i, q in enumerate(Q):
    lf = np.log(Fq2[i])
    sl = np.gradient(lf, log_s2)
    sl_s = np.convolve(sl, np.ones(3) / 3, mode="same")
    print(f"q={q:+.0f}: " + " ".join(f"{sl_s[j]:+.2f}@s{scales2[j]}"
                                     for j in range(0, len(scales2), 8)))
