"""Izmir serisinin saatlik hacmi: olay oncesi (rutin) pencere var mi?"""
import pandas as pd
import numpy as np

s = pd.read_csv(r"C:\TUBITAK_Proje\results\izmir2020_series_1min.csv",
                index_col=0, parse_dates=True)
print(f"N={len(s)}  aralik: {s.index.min()} -> {s.index.max()}")
print()
h = s["count"].resample("1h").sum()
for ts, v in h.items():
    bar = "#" * int(v / 2000)
    print(f"{ts:%m-%d %H:%M}  {int(v):>7,}  {bar}")

print()
print("=== OZET ===")
pre = s[s.index < pd.Timestamp("2020-10-30T11:51", tz="UTC")]
post = s[s.index >= pd.Timestamp("2020-10-30T11:51", tz="UTC")]
print(f"Olay oncesi: {len(pre)} dk, toplam tweet={int(pre['count'].sum()):,}, "
      f"ort={pre['count'].mean():.1f}/dk")
print(f"Olay sonrasi: {len(post)} dk, toplam tweet={int(post['count'].sum()):,}, "
      f"ort={post['count'].mean():.1f}/dk")
if len(pre):
    print(f"\nOlay oncesi distress ort={pre['distress'].mean():.4f} "
          f"std={pre['distress'].std():.4f}")
print(f"Olay sonrasi distress ort={post['distress'].mean():.4f} "
      f"std={post['distress'].std():.4f}")
