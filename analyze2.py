"""
Bir olayin 1-dk duygu serisinde, SECILEN sinyal kolonu icin MF-DFA + surrogate.

Kullanim:
  python analyze2.py <series_csv> <prefix> <signal> <acute_start_iso> [acute_hours] [n_surr]
Ornek:
  python analyze2.py results\km2023_series_1min.csv km2023 distress 2023-02-06T00:00
  python analyze2.py results\km2023_series_1min.csv km2023 sadness  2023-02-06T00:00

Pencereler: TUM | AKUT | NORM(son 48s) | HACIM(log tweet/dk, kontrol)
Cikti: results\{prefix}_{signal}_mfdfa.csv ve .png
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, h_spread, surrogate_test

Q = np.linspace(-5, 5, 21)
series_csv = sys.argv[1]
prefix = sys.argv[2]
signal = sys.argv[3]
acute_start = pd.Timestamp(sys.argv[4], tz="UTC")
acute_hours = float(sys.argv[5]) if len(sys.argv) > 5 else 24.0
n_surr = int(sys.argv[6]) if len(sys.argv) > 6 else 100

s = pd.read_csv(series_csv, index_col=0, parse_dates=True)
if signal not in s.columns:
    raise SystemExit(f"KOLON YOK: {signal}. Mevcut: {list(s.columns)}")

x_all = s[signal].values.astype(float)
print(f"{prefix} | sinyal={signal} | N={len(x_all)} | "
      f"{s.index.min()} -> {s.index.max()}", flush=True)

acute_end = acute_start + pd.Timedelta(hours=acute_hours)
norm_start = s.index.max() - pd.Timedelta(hours=48)

windows = {
    "TUM": (s.index, s[signal].values.astype(float)),
    "AKUT": (s.index[(s.index >= acute_start) & (s.index < acute_end)],
             s.loc[(s.index >= acute_start) & (s.index < acute_end), signal].values.astype(float)),
    "NORM": (s.index[s.index >= norm_start],
             s.loc[s.index >= norm_start, signal].values.astype(float)),
    "HACIM": (s.index, np.log1p(s["count"].values.astype(float))),
}

rows = []
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

for name, (idx, x) in windows.items():
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        print(f"  [{name}] cok kisa ({len(x)}), atlandi", flush=True)
        continue
    res = mfdfa(x, Q, order=2, n_scales=30)
    width = spectrum_width(res)
    hsp = h_spread(res)
    print(f"\n[{name}] N={len(x)} DA={width:.4f} Dh={hsp:.4f}", flush=True)

    st_i = surrogate_test(x, Q, n_surr=n_surr, kind="iaaft", seed=42)
    st_s = surrogate_test(x, Q, n_surr=n_surr, kind="shuffle", seed=43)
    print(f"   IAAFT   z={st_i['z']:+.2f} p={st_i['p_value']:.4f} "
          f"(null {st_i['null_mean']:.3f}+-{st_i['null_std']:.3f})", flush=True)
    print(f"   Shuffle z={st_s['z']:+.2f} p={st_s['p_value']:.4f} "
          f"(null {st_s['null_mean']:.3f}+-{st_s['null_std']:.3f})", flush=True)

    rows.append(dict(event=prefix, signal=signal, window=name, N=len(x),
                     delta_alpha=round(width, 4), delta_h=round(hsp, 4),
                     iaaft_mean=round(st_i["null_mean"], 4),
                     iaaft_sd=round(st_i["null_std"], 4),
                     iaaft_z=round(st_i["z"], 3), iaaft_p=st_i["p_value"],
                     shuffle_mean=round(st_s["null_mean"], 4),
                     shuffle_sd=round(st_s["null_std"], 4),
                     shuffle_z=round(st_s["z"], 3), shuffle_p=st_s["p_value"]))

    if name in ("TUM", "AKUT", "NORM"):
        axes[0, 0].plot(res["alpha"], res["f_alpha"], "o-", ms=3, label=name)
        axes[0, 1].plot(Q, res["h_q"], "o-", ms=3, label=name)
        axes[1, 0].loglog(res["scales"], res["Fq"][10], "o-", ms=3, label=name)

axes[0, 0].set_xlabel(r"$\alpha$"); axes[0, 0].set_ylabel(r"$f(\alpha)$")
axes[0, 0].set_title(f"Çoklu Fraktal Spektrum - {prefix} [{signal}]")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
axes[0, 1].set_xlabel("q"); axes[0, 1].set_ylabel("h(q)")
axes[0, 1].set_title("Genelleştirilmiş Hurst - h(q)")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
axes[1, 0].set_xlabel("s (ölçek)"); axes[1, 0].set_ylabel("F(q=0)")
axes[1, 0].set_title("Dalgalanma fonksiyonu"); axes[1, 0].grid(alpha=0.3)

# zaman serisi + fazlar
axes[1, 1].plot(s.index, s[signal], lw=0.4, color="navy")
axes[1, 1].axvspan(acute_start, acute_end, alpha=0.2, color="red", label="akut")
axes[1, 1].axvspan(norm_start, s.index.max(), alpha=0.2, color="green", label="norm")
axes[1, 1].set_title(f"{signal} serisi + fazlar")
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
pp = rf"C:\TUBITAK_Proje\results\{prefix}_{signal}_mfdfa.png"
plt.savefig(pp, dpi=140)
plt.close(fig)

out = pd.DataFrame(rows)
cp = rf"C:\TUBITAK_Proje\results\{prefix}_{signal}_mfdfa.csv"
out.to_csv(cp, index=False)
print("\n" + "=" * 74, flush=True)
print(out.to_string(index=False), flush=True)
print(f"\nKaydedildi: {pp}\n           {cp}", flush=True)
