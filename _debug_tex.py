"""LaTeX $ ve % denge analizi (xelatex oncesi)."""
import re

P = r"C:\TUBITAK_Proje\rapor\proje_detay.tex"
lines = open(P, encoding="utf-8").read().split("\n")

# 1) Bosluk/satir sonu disinda kalan $ sayisi
t = open(P, encoding="utf-8").read()
print("Toplam '$' sayisi:", t.count("$"), "(cift olmali)")

# 2) Her satirda kac '$' var (escape'li olanlar haric)
print("\n--- '$' sayisi tek olan satirlar (supheli) ---")
for i, ln in enumerate(lines, 1):
    # \\\$ kacisli olanlari temizle
    c = ln.replace("\\$", "")
    if c.count("$") % 2 == 1:
        print(f"  {i:5d}: {ln[:110]}")

# 3) '%' iceren satirlar
print("\n--- '%' iceren satirlar ---")
for i, ln in enumerate(lines, 1):
    if "%" in ln and not ln.lstrip().startswith("%"):
        print(f"  {i:5d}: {ln[:110]}")

# 4) escaped % (\%)
print("\n'\\%' sayisi:", t.count("\\%"))
