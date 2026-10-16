"""
Model Karsilastirma Testi v2 - sonuclari UTF-8 dosyaya yazar.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = r"C:\TUBITAK_Proje\results\model_comparison.txt"

PROBES = [
    ("Bugun hava cok guzel, cok mutluyum ve huzurluyum", "P"),
    ("Harika bir haber aldik, cok sevindim", "P"),
    ("Yardimlar bolgeye ulasti, emegi gecen herkese tesekkurler", "P"),
    ("Enkazdan bir kisi sag olarak kurtarildi, cok mutluyuz", "P"),
    ("Deprem cok korkunc, herkes perisan ve uzgun", "N"),
    ("Bu nasil bir felaket, yuregimiz yaniyor", "N"),
    ("Cok uzgunum, hicbir sey iyiye gitmiyor", "N"),
    ("Berbat bir gun gecirdim, sinirlerim cok bozuk", "N"),
    ("Deprem saat 04.17'de meydana geldi", "0"),
    ("Bolgede 10 il etkilendi ve hasar buyuk", "0"),
    ("AFAD konuyla ilgili aciklama yapti", "0"),
    ("Yarin toplanti saat 10'da yapilacak", "0"),
]

MODELS = [
    "savasy/bert-base-turkish-sentiment-cased",
    "anilguven/bert_tr_turkish_tweet",
    "saribasmetehan/bert-base-turkish-sentiment-analysis",
    "furkankarakuz/turkish-emotion-classifier",
]

lines = []
lines.append("MODEL KARSILASTIRMA RAPORU")
lines.append("=" * 78)


def w(s=""):
    lines.append(s)
    print(s.encode("ascii", "replace").decode("ascii"))


for model_name in MODELS:
    w("=" * 78)
    w(f"MODEL: {model_name}")
    w("=" * 78)
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        model.eval()
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        w(f"Etiketler: {id2label}")

        with torch.no_grad():
            enc = tok([p[0] for p in PROBES], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(DEVICE)
            probs = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()

        for (text, exp), p in zip(PROBES, probs):
            vec = " ".join(f"{x:.2f}" for x in p)
            w(f"  beklenen={exp} | P=[{vec}] | {text[:50]}")

    except Exception as e:
        w(f"  HATA: {type(e).__name__}: {str(e)[:200]}")
    w()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nRapor kaydedildi: {OUT}")
