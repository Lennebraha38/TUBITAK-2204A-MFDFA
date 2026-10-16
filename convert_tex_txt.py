"""Hizli LaTeX -> duz metin cevirici (proje_detay.tex -> proje_detay.txt)."""
import re

SRC = r"C:\TUBITAK_Proje\rapor\proje_detay.tex"
DST = r"C:\TUBITAK_Proje\rapor\proje_detay.txt"

t = open(SRC, encoding="utf-8").read()

# sadece govde
t = t[t.index(r"\begin{document}") + len(r"\begin{document}"):]
t = t[:t.index(r"\end{document}")]

# yorumlari temizle (\% korunur)
t = t.replace(r"\%", "\x00")
t = re.sub(r"(?<!\\)%.*", "", t)
t = t.replace("\x00", "%")

# basit komutlar
t = t.replace(r"\maketitle", "")
t = t.replace(r"\tableofcontents",
              "\n[ İÇİNDEKİLER - Overleaf'te otomatik oluşur ]\n")
t = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", t)
t = t.replace(r"\newpage", "\n" + "=" * 78 + "\n")
t = t.replace(r"\vspace{0.5cm}", "")
t = re.sub(r"\\vspace\{[^}]*\}", "", t)
t = re.sub(r"\\thispagestyle\{[^}]*\}", "", t)

# basliklar
t = re.sub(r"\\section\*\{([^{}]*)\}", r"\n\n\n" + "#" * 70 + r"\n###  \1\n" + "=" * 70 + "\n", t)
t = re.sub(r"\\section\{([^}]*)\}",     r"\n\n\n===== \1 =====\n", t)
t = re.sub(r"\\subsection\{([^}]*)\}",  r"\n\n--- \1 ---\n", t)
t = re.sub(r"\\subsubsection\{([^}]*)\}", r"\n\n" + "\u00b7" + r" \1\n", t)

# addcontentsline / label
t = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", t)
t = re.sub(r"\\label\{[^}]*\}", "", t)
t = re.sub(r"\\ref\{[^}]*\}", "", t)
t = re.sub(r"\\cite\{[^}]*\}", "", t)

# listeler / ortamlar
t = re.sub(r"\\begin\{(itemize|enumerate|description)\}\[[^\]]*\]", "", t)
t = re.sub(r"\\begin\{(itemize|enumerate|description)\}", "", t)
t = re.sub(r"\\end\{(itemize|enumerate|description)\}", "", t)
t = re.sub(r"\\item\[([^\]]*)\]", r"\n  [\1] ", t)
t = re.sub(r"\\item", "\n  - ", t)
t = re.sub(r"\\begin\{(center|quote|minipage)\}\{[^}]*\}", "", t)
t = re.sub(r"\\begin\{(center|quote|minipage)\}", "", t)
t = re.sub(r"\\end\{(center|quote|minipage)\}", "", t)
t = re.sub(r"\\begin\{(table|figure)\}\[[^\]]*\]", "", t)
t = re.sub(r"\\begin\{(table|figure)\}", "", t)
t = re.sub(r"\\end\{(table|figure)\}", "", t)
t = re.sub(r"\\caption\{([^}]*)\}", r"\n  [\1]\n", t)
t = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", t)
t = re.sub(r"\\end\{tabular\}", "", t)
t = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "-" * 60, t)
t = re.sub(r"\\cmidrule\(?[^)]*\)?\{[^}]*\}", "", t)
t = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", "\nKAYNAKLAR\n", t)
t = re.sub(r"\\end\{thebibliography\}", "", t)
t = re.sub(r"\\begin\{document\}", "", t)
t = re.sub(r"\\end\{document\}", "", t)
t = re.sub(r"\\vspace\{[^}]*\}", "", t)
t = re.sub(r"\\appendix", "", t)

# denklemler
t = re.sub(r"\\begin\{(equation|align)\*?\}", "", t)
t = re.sub(r"\\end\{(equation|align)\*?\}", "", t)

# tablo duzeni: multicolumn, satir sonu, hücre ayraci
t = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", t)
t = re.sub(r"\\cmidrule\(?[^)]*\)?\{[^}]*\}", "", t)
t = t.replace("\\\\", "\n")
t = t.replace("&", " | ")

# metin bicimleri
t = re.sub(r"\\(textbf|textit|emph|texttt|text|mathrm|mathbf|underline)\{([^{}]*)\}", r"\2", t)
# bibitem -> [n]
def _bib(m):
    return f"\n[{m.group(2)}] "
t = re.sub(r"\\bibitem\{([^}]*)\}", lambda m: "\n\n[*] ", t)

# matematik sembolleri
MATH = {
    r"\Dal": "Delta_alpha", r"\Dal": "Delta_alpha",
    r"\Delta\alpha": "Delta_alpha", r"\Delta": "Delta",
    r"\alpha": "alpha", r"\beta": "beta", r"\tau": "tau",
    r"\phi": "phi", r"\sigma": "sigma", r"\mu": "mu",
    r"\sim": "~", r"\to": "->", r"\approx": "~=", r"\times": "x",
    r"\cdot": ".", r"\le": "<=", r"\ge": ">=", r"\neq": "!=",
    r"\infty": "sonsuz", r"\sum": "Toplam", r"\int": "Int",
    r"\ln": "ln", r"\log": "log", r"\exp": "exp", r"\prod": "Carpim",
    r"\min": "min", r"\max": "max", r"\lim": "lim",
    r"\lfloor": "[", r"\rfloor": "]", r"\lceil": "[", r"\rceil": "]",
    r"\bigl": "", r"\bigr": "", r"\Bigl": "", r"\Bigr": "",
    r"\left": "", r"\right": "", r"\big": "", r"\Big": "",
    r"\displaystyle": "", r"\dots": "...", r"\ldots": "...",
    r"\%": "%", r"\&": "&", r"\_": "_", r"\ ": " ", r"\,": " ",
    r"\{\}": "", r"\;": " ", r"\quad": "    ", r"\qquad": "        ",
}
for k, v in sorted(MATH.items(), key=lambda x: -len(x[0])):
    t = t.replace(k, v)
# sinirlar: ^{...} _{...}
t = re.sub(r"\^\{([^}]*)\}", r"^\1", t)
t = re.sub(r"_\{([^}]*)\}", r"_\1", t)
# frac
t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", t)
t = re.sub(r"\\sqrt\{([^}]*)\}", r"karekök(\1)", t)
# kalan $ isaretleri
t = t.replace("$", "")
# kalan ters egik cizgili komutlar
t = re.sub(r"\\[a-zA-Z]+", "", t)
t = t.replace("{", "").replace("}", "")

# tirnaklar / tireler
t = t.replace("''", '"').replace("``", '"')
t = t.replace("---", "—").replace("--", "–")
t = t.replace("~", " ")

# bosluk temizligi
t = re.sub(r"(\w) '", r"\1'", t)
t = re.sub(r"\( ", "(", t)
t = re.sub(r" \)", ")", t)
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r" *\n *", "\n", t)
t = re.sub(r"\n{3,}", "\n\n", t)

# baslik + not
head = (
    "SOSYAL MEDYA DUYGU ZAMAN SERİLERİNİN ÇOKLU FRAKTAL ANALİZİ İLE\n"
    "KAOTİK VE STOKASTİK DİNAMİKLERİN AYRIŞTIRILMASI\n"
    "TÜBİTAK 2204-A Proje Detay Raporu (düz metin sürümü)\n"
    + "=" * 72 + "\n\n"
)
open(DST, "w", encoding="utf-8").write(head + t.strip() + "\n")

txt = head + t
print("Yazildi:", DST)
print("Karakter:", len(txt), "| Kelime:", len(txt.split()), "| Satir:", txt.count(chr(10)))
