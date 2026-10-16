"""Izmir olay anini dakika cozunurlugunde dogrula."""
import pandas as pd

s = pd.read_csv(r"C:\TUBITAK_Proje\results\izmir2020_series_1min.csv",
                index_col=0, parse_dates=True)
w = s.loc["2020-10-30 10:00":"2020-10-30 13:00", "count"]
print("Dakika-bazli tweet sayisi (UTC):")
run = 0
for ts, v in w.items():
    if v > 0:
        run = 0
    else:
        run += 1
    if v > 50 or (run <= 3):
        print(f"  {ts:%H:%M}  {int(v):>5d}")
# ilk buyuk artis
big = s[s["count"] > 200]
print(f"\nIlk count>200: {big.index.min()}")
print(f"Toplam bos kutu: {(s['count']==0).sum()} / {len(s)}")
pre = s.loc[:"2020-10-30 11:50"]
print(f"11:50 oncesi bos kutu: {(pre['count']==0).sum()} / {len(pre)}")
