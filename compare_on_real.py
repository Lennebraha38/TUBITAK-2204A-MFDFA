"""
Farkli duygu modellerini GERCEK deprem tweetlerinde karsilastir.
Amac: en tutarli (saglikli dagilim veren) modeli secmek.
Sonuclari UTF-8 dosyaya yazar.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda"
N = 3000
orig = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv",
                   usecols=["Tweet"])
rng = np.random.default_rng(11)
idx = rng.choice(len(orig), size=N, replace=False)
texts = [str(orig["Tweet"].iloc[i]) for i in idx]

lines = []
def w(s=""):
    lines.append(s)
    print(s.encode("ascii", "replace").decode("ascii"), flush=True)


@torch.inference_mode()
def score(model_name, bs=256, maxlen=64):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval().half()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=maxlen, return_tensors="pt").to(DEVICE)
        out.append(F.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return id2label, np.vstack(out)


def report(name, id2label, P):
    w("=" * 78)
    w(f"MODEL: {name}")
    w(f"Etiketler: {id2label}")
    am = P.argmax(1)
    for j, lab in id2label.items():
        w(f"  {lab:12s}: ort={P[:,j].mean():.3f}  argmax={(am==j).mean()*100:5.1f}%")
    w()


for mn in ["savasy/bert-base-turkish-sentiment-cased",
           "saribasmetehan/bert-base-turkish-sentiment-analysis",
           "furkankarakuz/turkish-emotion-classifier"]:
    try:
        id2label, P = score(mn)
        report(mn, id2label, P)
    except Exception as e:
        w(f"MODEL HATASI {mn}: {type(e).__name__}: {str(e)[:150]}")

# En iyi aday: her model icin "en negatif" ve "en pozitif" 8 ornek
w("#" * 78)
w("ORNEK INCELEME (savasy binary)")
w("#" * 78)
try:
    id2label, P = score("savasy/bert-base-turkish-sentiment-cased")
    v = P[:, 1] - P[:, 0]  # pos - neg
    order = np.argsort(v)
    w("-- en NEGATIF 8 --")
    for i in order[:8]:
        w(f"[{v[i]:+.2f}] {texts[i][:110]}")
    w("-- en POZITIF 8 --")
    for i in order[-8:]:
        w(f"[{v[i]:+.2f}] {texts[i][:110]}")
except Exception as e:
    w(f"HATA: {e}")

out = r"C:\TUBITAK_Proje\results\real_model_compare.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nYazildi:", out, flush=True)
