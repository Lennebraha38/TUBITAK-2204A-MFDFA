"""
Model Karsilastirma Testi
Amaç: En iyi kalibre edilmis Turkce duygu modelini secmek.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test cumleleri: (metin, beklenen)  beklenen: P / N / 0
PROBES = [
    ("Bugun hava cok guzel, cok mutluyum ve huzurluyum", "P"),
    ("Harika bir haber aldik, cok sevindim", "P"),
    ("Yardimlar bolgeye ulasti, emegi gecen herkese tesekkurler", "P"),
    ("Enkazdan bir kisi sag olarak kurtarildi, cok mutluyuz", "P"),
    ("Deprem cok korkunctу, herkes perisan ve uzgun", "N"),
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
]


def valence(probs, id2label):
    """P(pozitif) - P(negatif) seklinde skor uret."""
    pos = neg = 0.0
    for i, name in id2label.items():
        n = name.lower()
        if "pos" in n or "lab_1" in n or n == "1":
            pos = float(probs[i])
        if "neg" in n or "lab_0" in n or n == "0":
            neg = float(probs[i])
    return pos - neg


for model_name in MODELS:
    print("=" * 72)
    print(f"MODEL: {model_name}")
    print("=" * 72)
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        model.eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        print("Etiketler:", id2label)

        scores = []
        with torch.no_grad():
            enc = tok([p[0] for p in PROBES], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(DEVICE)
            probs = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for (text, exp), p in zip(PROBES, probs):
            v = valence(p, id2label)
            if v > 0.1:
                pred = "P"
            elif v < -0.1:
                pred = "N"
            else:
                pred = "0"
            ok = "OK " if pred == exp else "HATA"
            print(f"  [{v:+.3f}] {pred}/{exp} {ok} | {text[:52]}")

        # Basari: sadece net P/N cumlelerde
        correct = 0
        total = 0
        for (text, exp), p in zip(PROBES, probs):
            v = valence(p, id2label)
            pred = "P" if v > 0.1 else ("N" if v < -0.1 else "0")
            if exp in ("P", "N"):
                total += 1
                if pred == exp:
                    correct += 1
        print(f"\n  Net P/N basari: {correct}/{total} = {correct/total*100:.0f}%")

    except Exception as e:
        print(f"  HATA: {type(e).__name__}: {str(e)[:150]}")
    print()
