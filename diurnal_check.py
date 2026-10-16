"""
Sirkadiyen (24s) dongu etkisi: duygu sinyalinde gunluk ritim var mi,
ve coklu fraktalligi etkiliyor mu?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, surrogate_test

Q = np.linspace(-5, 5, 21)


def analyze(label, ser):
    x = ser.values.astype(float)
    ok = np.isfinite(x)
    x = x[ok]
    idx = ser.index[ok]
    r = mfdfa(x, Q)
    print(f"\n### {label} (N={len(x)})")
    print(f"  ham           DA={spectrum_width(r):.4f}")

    # saatlik ortalama cikar (sirkadiyen)
    hr = idx.hour
    prof = pd.Series(x).groupby(hr).transform("mean").values
    xd = x - prof
    rd = mfdfa(xd, Q)
    print(f"  saatlik-ort-cikarildi DA={spectrum_width(rd):.4f}")

    # 24s hareketli ortalama cikar
    xm = x - pd.Series(x).rolling(1441, center=True, min_periods=1).mean().values
    rm = mfdfa(xm, Q)
    print(f"  24s-har-ort-cikarildi  DA={spectrum_width(rm):.4f}")

    # saatlik profil genligi
    print(f"  saatlik distress profili: min={prof.min() if hasattr(prof,'min') else 0:.3f} "
          f"max={prof.max() if hasattr(prof,'max') else 0:.3f} "
          f"genlik={prof.max()-prof.min():.3f} std={prof.std():.3f}")
    print(f"  genel std={x.std():.4f}")
    # IAAFT z (ham)
    st = surrogate_test(x, Q, n_surr=50, kind="iaaft", seed=1)
    print(f"  ham IAAFT z={st['z']:+.2f} p={st['p_value']:.4f}")


s1 = pd.read_csv(r"C:\TUBITAK_Proje\results\km2023_series_1min.csv",
                 index_col=0, parse_dates=True)
s2 = pd.read_csv(r"C:\TUBITAK_Proje\results\izmir2020_series_1min.csv",
                 index_col=0, parse_dates=True)

analyze("km2023 distress", s1["distress"])
analyze("izmir2020 distress", s2["distress"])

print("\n=== Saatlik distress profilleri ===")
for lbl, s in [("km2023", s1), ("izmir2020", s2)]:
    p = s["distress"].groupby(s.index.hour).mean()
    print(f"{lbl}: " + " ".join(f"{h:02d}:{v:.2f}" for h, v in p.items()))
