"""Yuksek pozitif/negatif skorlu GERCEK tweetleri ornekle ve incele."""
import pandas as pd
import numpy as np

orig = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv",
                   usecols=["Tweet"])
sc = pd.read_csv(r"C:\TUBITAK_Proje\results\scored\depremGun1tr_scored.csv.gz")
assert len(orig) == len(sc), (len(orig), len(sc))

rng = np.random.default_rng(7)
lines = []
lines.append("=" * 80)
lines.append("YUKSEK POZITIF (p_pos>0.95) - 25 ornek")
lines.append("=" * 80)
idx = np.where(sc["p_pos"].values > 0.95)[0]
for i in rng.choice(idx, size=min(25, len(idx)), replace=False):
    t = str(orig["Tweet"].iloc[i]).replace("\n", " ")
    lines.append(f"[{sc['p_pos'].iloc[i]:.2f}] {t[:150]}")

lines.append("")
lines.append("=" * 80)
lines.append("YUKSEK NEGATIF (p_neg>0.95) - 25 ornek")
lines.append("=" * 80)
idx = np.where(sc["p_neg"].values > 0.95)[0]
for i in rng.choice(idx, size=min(25, len(idx)), replace=False):
    t = str(orig["Tweet"].iloc[i]).replace("\n", " ")
    lines.append(f"[{sc['p_neg'].iloc[i]:.2f}] {t[:150]}")

lines.append("")
lines.append("=" * 80)
lines.append("YUKSEK NOTR (p_neu>0.9) - 15 ornek")
lines.append("=" * 80)
idx = np.where(sc["p_neu"].values > 0.9)[0]
for i in rng.choice(idx, size=min(15, len(idx)), replace=False):
    t = str(orig["Tweet"].iloc[i]).replace("\n", " ")
    lines.append(f"[{sc['p_neu'].iloc[i]:.2f}] {t[:150]}")

out = r"C:\TUBITAK_Proje\results\sample_inspect.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Yazildi:", out)
