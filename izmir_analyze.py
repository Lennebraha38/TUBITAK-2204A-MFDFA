"""Izmir deprem verisinin yapisini ve hacmini analiz et."""
import pandas as pd
import numpy as np

SRC = r"C:\TUBITAK_Proje\data\izmir\izmir_earthquake.csv"
print("Okunuyor (sadece Datetime)...", flush=True)
dt = pd.read_csv(SRC, usecols=["Datetime"])
print(f"Toplam satir: {len(dt):,}", flush=True)
dt["Datetime"] = pd.to_datetime(dt["Datetime"], utc=True, errors="coerce")
bad = dt["Datetime"].isna().sum()
print(f"Gecersiz tarih: {bad}", flush=True)
d = dt.dropna().set_index("Datetime").sort_index()
print(f"Aralik: {d.index.min()} -> {d.index.max()}", flush=True)
print(f"Sure: {d.index.max() - d.index.min()}", flush=True)

hourly = d.resample("1h").size()
print("\n=== SAATLIK HACIM (ilk 100 saat) ===", flush=True)
# Deprem: 30 Ekim 2020 14:51 TR (11:51 UTC)
for ts, v in list(hourly.items())[:100]:
    bar = "#" * int(v / 200)
    print(f"{ts.strftime('%d %b %H:%M')} | {v:>6,} {bar}", flush=True)

print("\n=== GUNLUK ===", flush=True)
for ts, v in d.resample("1D").size().items():
    print(f"{ts.strftime('%d %b')} | {v:>8,}", flush=True)

# Dakika bazli yogunluk (ilk 6 saat)
mx = d.index.min()
first6 = d[(d.index >= mx) & (d.index < mx + pd.Timedelta(hours=6))]
mins = first6.resample("1min").size()
print(f"\nIlk 6 saatte: {len(first6):,} tweet, {len(mins)} dakika, "
      f"doluluk: {(mins>0).mean()*100:.0f}%", flush=True)
print(f"Dakika basi: ort={mins.mean():.1f} med={mins.median():.0f} max={mins.max()}", flush=True)

# Tum veride dakika dolulugu
allmin = d.resample("1min").size()
span = int((d.index.max() - d.index.min()).total_seconds() / 60)
print(f"\nTum aralikta dakika: {span}, dolu: {(allmin>0).sum()} "
      f"({(allmin>0).mean()*100:.0f}%)", flush=True)
