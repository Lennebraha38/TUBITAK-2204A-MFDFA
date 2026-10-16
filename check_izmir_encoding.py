"""Izmir CSV encoding teshisi."""
p = r"C:\TUBITAK_Proje\data\izmir\izmir_earthquake.csv"
raw = open(p, "rb").read(400)
lines = []
lines.append("HAM BAYTLAR (ilk 200):")
lines.append(repr(raw[:200]))
lines.append("")
for enc in ["utf-8", "cp1254", "latin-1", "utf-8-sig"]:
    try:
        s = raw.decode(enc)
        lines.append(f"[{enc:10s}] OK: {s[:90]!r}")
    except Exception as e:
        lines.append(f"[{enc:10s}] HATA: {type(e).__name__}: {str(e)[:80]}")

# pandas ile bir kac satir
import pandas as pd
for enc in ["utf-8", "cp1254"]:
    try:
        df = pd.read_csv(p, usecols=["Text"], nrows=3, encoding=enc)
        lines.append(f"\npandas[{enc}]: {df['Text'].iloc[0][:80]!r}")
    except Exception as e:
        lines.append(f"\npandas[{enc}] HATA: {str(e)[:100]}")

out = r"C:\TUBITAK_Proje\results\izmir_encoding_check.txt"
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("Yazildi:", out)
