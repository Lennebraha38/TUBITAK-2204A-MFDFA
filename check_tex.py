"""
LaTeX butunluk denetimi (XeLaTeX oncesi statik kontrol).

Kontrol edilenler:
  - \\begin / \\end dengei
  - \\label / \\ref eslesmesi (tanimsiz referans var mi?)
  - Etiket adlarinda ASCII disi karakter var mi? (XeLaTeX riski)
  - \\appendix birden fazla cagriliyor mu?
  - \\section* / \\section sayimi
  - Yuzde isareti: matematik modunda (\\$...\\$) kullanilmis mi? (LaTeX hatasi)
  - Tirnak isareti: ``...'' kapatilmis mi?
  - Kelime sayisi (rapor uzunlugu)

Kullanim: python check_tex.py [dosya]
"""
import collections
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\TUBITAK_Proje\rapor\proje_detay.tex"
t = open(PATH, encoding="utf-8").read()
ok = True


def chk(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'OK ' if cond else 'HATA'}] {label}" + (f"  -> {detail}" if detail else ""))


print(f"Denetlenen dosya: {PATH}")
print(f"Boyut: {len(t):,} karakter\n")

# --- 1. begin/end dengesi -------------------------------------------------
beg = collections.Counter(re.findall(r"\\begin\{([^}]+)\}", t))
end = collections.Counter(re.findall(r"\\end\{([^}]+)\}", t))
bad = {k: (beg[k], end[k]) for k in set(beg) | set(end) if beg[k] != end[k]}
chk("begin/end dengesi", not bad, str(bad) if bad else f"{sum(beg.values())} ortam")

# --- 2. label/ref eslesmesi ----------------------------------------------
labs = set(re.findall(r"\\label\{([^}]+)\}", t))
refs = set(re.findall(r"\\(?:eq|page|auto|c)?ref\{([^}]+)\}", t))
chk("tum \\ref tanimli", not (refs - labs), f"tanimsiz: {sorted(refs - labs)}")
chk("tum \\label kullanilmis", not (labs - refs), f"kullanilmayan: {sorted(labs - refs)}")

# --- 3. etiket adlari ASCII ----------------------------------------------
non_ascii = [l for l in labs if any(ord(c) > 127 for c in l)]
chk("etiket adlari ASCII", not non_ascii, str(non_ascii))

# --- 4. appendix ----------------------------------------------------------
n_app = len(re.findall(r"\\appendix", t))
chk("appendix en fazla 1 kez", n_app <= 1, f"{n_app} kez")

# --- 5. bolum sayisi ------------------------------------------------------
n_sec = len(re.findall(r"\\section\*?\{", t))
chk("bolum sayisi makul", 5 <= n_sec <= 40, f"{n_sec} bolum")

# --- 6. matematik modunda yuzde -------------------------------------------
# Once LaTeX yorum satirlarini temizle: satir basindaki % yorumdur.
def _strip_comments(txt):
    out = []
    for ln in txt.split("\n"):
        # kacisli % (\\%) degildir; ilk gercek % yorum baslatir
        j = 0
        while True:
            k = ln.find("%", j)
            if k < 0:
                break
            if k > 0 and ln[k - 1] == "\\":
                j = k + 1
                continue
            ln = ln[:k]
            break
        out.append(ln)
    return "\n".join(out)


t_nc = _strip_comments(t)
math_pct = re.findall(r"\$[^$]*%[^$]*\$", t_nc)
chk("matematik modunda % yok", not math_pct, str(math_pct[:3]))

# 6b. kacisli yuzde sayisi (dengeli olmali)
n_pct_esc = len(re.findall(r"\\%", t_nc))
chk("kacisli yuzde dengeli", n_pct_esc % 2 == 0, f"{n_pct_esc} adet")

# --- 7. tirnaklar ---------------------------------------------------------
n_open = len(re.findall(r"``", t))
n_close = len(re.findall(r"''", t))
chk("tirnaklar dengeli", n_open == n_close, f"`` ={n_open}, '' ={n_close}")

# --- 8. metrik ------------------------------------------------------------
words = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", t))
print(f"\n  Kelime sayisi (yaklasik): {words:,}")

print("\nSONUC:", "TUM DENETIMLER GECTI" if ok else "HATA VAR")
sys.exit(0 if ok else 1)
