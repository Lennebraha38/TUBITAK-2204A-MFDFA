"""
Bir olayin 1-dk duygu serisinde MF-DFA + surrogate testleri.
Kullanim: python analyze_series.py <series_csv> <prefix> <acute_start_iso>
Ornek   : python analyze_series.py results\km2023_series_1min.csv km2023 2023-02-06T00:00
"""
import sys
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
acute_start = pd.Timestamp(sys.argv[3], tz="UTC")
if len(sys.argv) > 4:
    acute_hours = float(sys.argv[4])
else:
    acute_hours = 24.0

s = pd.read_csv(series_csv, parse_dates=["Date"]).set_index("Date")
x_all = s["valence"].values.astype(float)
print(f"{prefix}: N={len(x_all)} {s.index.min()} -> {s.index.max()}", flush=True)

# Pencereler
acute_end = acute_start + pd.Timedelta(hours=acute_hours)
norm_start = s.index.max() - pd.Timedelta(hours=48)

windows = {
    "TUM": s["valence"],
    "AKUT_24s": s.loc[(s.index >= acute_start) & (s.index < acute_end), "valence"],
    "NORM_son48s": s.loc[s.index >= norm_start, "valence"],
}
# Hacim serisi (log) - ikinci analiz icin
vol = np.log1p(s["count"].values.astype(float))
windows["HACIM_log_TUM"] = pd.Series(vol, index=s.index)

rows = []
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

for name, ser in windows.items():
    x = ser.values.astype(float)
    if len(x) < 64:
        print(f"  [{name}] cok kisa, atlandi", flush=True)
        continue
    res = mfdfa(x, Q, order=2, n_scales=30)
    width = spectrum_width(res)
    hsp = h_spread(res)
    print(f"\n[{name}] N={len(x)} Delta_alpha={width:.4f} Delta_h={hsp:.4f}", flush=True)

    st_i = surrogate_test(x, Q, n_surr=100, kind="iaaft", seed=42)
    st_s = surrogate_test(x, Q, n_surr=100, kind="shuffle", seed=43)
    print(f"   IAAFT   z={st_i['z']:.2f} p={st_i['p_value']:.4f} "
          f"(null {st_i['null_mean']:.3f}+-{st_i['null_std']:.3f})", flush=True)
    print(f"   Shuffle z={st_s['z']:.2f} p={st_s['p_value']:.4f}", flush=True)

    rows.append(dict(event=prefix, window=name, N=len(x), width=width,
                     hspread=hsp,
                     iaaft_mean=st_i["null_mean"], iaaft_sd=st_i["null_std"],
                     iaaft_z=st_i["z"], iaaft_p=st_i["p_value"],
                     shuffle_mean=st_s["null_mean"], shuffle_sd=st_s["null_std"],
                     shuffle_z=st_s["z"], shuffle_p=st_s["p_value"]))

    if name.startswith(("TUM", "AKUT", "NORM")):
        axes[0, 0].plot(res["alpha"], res["f_alpha"], "o-", ms=3, label=name)
        axes[0, 1].plot(Q, res["h_q"], "o-", ms=3, label=name)
        axes[1, 0].loglog(res["scales"], res["Fq"][10], "o-", ms=3, label=name)

axes[0, 0].set_xlabel(r"$\alpha$"); axes[0, 0].set_ylabel(r"$f(\alpha)$")
axes[0, 0].set_title(f"Çoklu Fraktal Spektrum - {prefix}")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
axes[0, 1].set_xlabel("q"); axes[0, 1].set_ylabel("h(q)")
axes[0, 1].set_title("Genelleştirilmiş Hurst - h(q)")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
axes[1, 0].set_xlabel("s (ölçek)"); axes[1, 0].set_ylabel("F(q=0)")
axes[1, 0].set_title("Dalgalanma fonksiyonu"); axes[1, 0].grid(alpha=0.3)

# Valans serisi
axes[1, 1].plot(s.index, s["valence"], lw=0.4, color="navy")
axes[1, 1].axhline(0, color="k", lw=0.5)
axes[1, 1].axvspan(acute_start, acute_end, alpha=0.2, color="red", label="akut")
axes[1, 1].axvspan(norm_start, s.index.max(), alpha=0.2, color="green", label="norm")
axes[1, 1].set_title("Valans serisi + fazlar"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
pp = rf"C:\TUBITAK_Proje\results\{prefix}_mfdfa.png"
plt.savefig(pp, dpi=140)

out = pd.DataFrame(rows)
cp = rf"C:\TUBITAK_Proje\results\{prefix}_mfdfa.csv"
out.to_csv(cp, index=False)
print("\n" + "=" * 70, flush=True)
print(out.to_string(index=False), flush=True)
print(f"\nKaydedildi: {pp}\n           {cp}", flush=True)
