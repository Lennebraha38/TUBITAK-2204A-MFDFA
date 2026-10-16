"""
Tum deprem tweetlerini skorla -> valans + olasiliklar (OPTIMIZE: FP16, bs512, len64)
Model: saribasmetehan/bert-base-turkish-sentiment-analysis
  LABEL_1 = Positive, LABEL_0 = Notr, LABEL_2 = Negative
Her dosya ayri kaydedilir (kesintiye dayanikli).
"""
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "saribasmetehan/bert-base-turkish-sentiment-analysis"
DEVICE = "cuda"
BASE = r"C:\TUBITAK_Proje\data\earthquake\turkish"
OUTDIR = r"C:\TUBITAK_Proje\results\scored"
os.makedirs(OUTDIR, exist_ok=True)
FILES = ['depremGun1tr.csv', 'depremGun2tr.csv', 'depremGun3tr.csv',
         'depremGun4tr.csv', 'depremGun5tr.csv']
L_POS, L_NEU, L_NEG = 1, 0, 2

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)
model.eval()
model.half()
print(f"Model hazir (fp16). id2label={model.config.id2label}", flush=True)

url_re = re.compile(r"http\S+|www\.\S+")
mention_re = re.compile(r"@\w+")
space_re = re.compile(r"\s+")


def clean(t):
    t = str(t)
    t = url_re.sub(" ", t)
    t = mention_re.sub(" ", t)
    t = t.replace("RT ", " ")
    return space_re.sub(" ", t).strip()


@torch.inference_mode()
def score_batch(texts, bs=512):
    n = len(texts)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(0, n, bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        out[i:i + len(chunk), 0] = probs[:, L_POS]
        out[i:i + len(chunk), 1] = probs[:, L_NEU]
        out[i:i + len(chunk), 2] = probs[:, L_NEG]
    return out


t_all = time.time()
for f in FILES:
    out_path = os.path.join(OUTDIR, f.replace(".csv", "_scored.csv.gz"))
    if os.path.exists(out_path):
        print(f"ATLANDI (zaten var): {out_path}", flush=True)
        continue

    t0 = time.time()
    df = pd.read_csv(os.path.join(BASE, f), usecols=["Date", "Tweet"])
    df["clean"] = df["Tweet"].map(clean)
    empty = (df["clean"].str.len() == 0).values
    texts = df["clean"].tolist()

    P = np.zeros((len(df), 3), dtype=np.float32)
    P[:, 1] = 1.0
    if (~empty).any():
        idx = np.where(~empty)[0]
        scored = score_batch([texts[i] for i in idx])
        P[idx] = scored

    out = pd.DataFrame({
        "Date": df["Date"],
        "valence": P[:, 0] - P[:, 2],
        "p_pos": P[:, 0], "p_neu": P[:, 1], "p_neg": P[:, 2],
    })
    out.to_csv(out_path, index=False, compression="gzip")
    el = time.time() - t0
    print(f"  {f}: {len(df):,} tweet | {el:.0f}s ({len(df)/el:.0f}/sn) -> {out_path}",
          flush=True)

print(f"\nTOPLAM SURE: {(time.time()-t_all)/60:.1f} dk", flush=True)
print("TUM DOSYALAR BITTI", flush=True)
