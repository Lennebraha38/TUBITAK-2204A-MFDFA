"""
Her q icin log F(q,s) vs log s dogrusal mi? R^2 kontrolu.
Boylekce hem protokol hem scaling bolgesi netlesir.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, pmodel_cascade


def fit_report(x, q_vals, s_min, s_max_frac, order=2, n_scales=30, label=""):
    x = np.asarray(x, float)
    N = len(x)
    Y = np.cumsum(x - x.mean())
    s_max = N // s_max_frac
    scales = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max),
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

    log_s = np.log(scales)
    r2s = []
    hs = []
    for i in range(len(q_vals)):
        lf = np.log(Fq[i])
        m = np.isfinite(lf)
        c = np.polyfit(log_s[m], lf[m], 1)
        pred = np.polyval(c, log_s[m])
        ss_res = np.sum((lf[m] - pred) ** 2)
        ss_tot = np.sum((lf[m] - lf[m].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot)
        hs.append(c[0])
    hs = np.array(hs)
    da = hs[0] - hs[-1]
    return dict(label=label, n_scales_scales=len(scales), s_range=(scales[0], scales[-1]),
                r2_mean=float(np.mean(r2s)), r2_min=float(np.min(r2s)),
                h_min=float(hs.min()), h_max=float(hs.max()), delta_h=da)


s = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                index_col=0, parse_dates=True)
x = s["distress"].values.astype(float)
ref = pmodel_cascade(8192, 0.3)
Q = np.linspace(-5, 5, 21)

print("=== R^2 ve Delta_alpha (s_min / s_max_frac kombinasyonlari) ===")
print(f"{'veri':10s} {'s_min':>6s} {'smf':>4s} {'s_aralik':>14s} "
      f"{'R2_ort':>7s} {'R2_min':>7s} {'DA':>8s}")
print("-" * 66)
for lbl, data in [("distress", x), ("p-model", ref)]:
    for smin in (16, 24, 32, 48, 64):
        for smf in (4, 8):
            r = fit_report(data, Q, smin, smf, label=lbl)
            print(f"{lbl:10s} {smin:6d} {smf:4d} "
                  f"{str(r['s_range']):>14s} {r['r2_mean']:7.4f} "
                  f"{r['r2_min']:7.4f} {r['delta_h']:8.4f}")
