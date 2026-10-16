"""
Izmir 2020 depremi tweetlerini skorla -> valans + olasiliklar.
Girdi : data/izmir/izmir_earthquake.csv (kolonlar: Datetime, Text, ...)
Cikti : results/scored/izmir_scored.csv.gz
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
SRC = r"C:\TUBITAK_Proje\data\izmir\izmir_earthquake.csv"
OUT = r"C:\TUBITAK_Proje\results\scored\izmir_scored.csv.gz"
L_POS, L_NEU, L_NEG = 1, 0, 2

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)
model.eval()
model.half()
print(f"Model hazir (fp16).", flush=True)

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


t0 = time.time()
print("Okunuyor (Datetime, Text, rescue)...", flush=True)
df = pd.read_csv(SRC, usecols=["Datetime", "Text", "rescue"])
print(f"  {len(df):,} satir", flush=True)

df["clean"] = df["Text"].map(clean)
empty = (df["clean"].str.len() == 0).values
texts = df["clean"].tolist()

P = np.zeros((len(df), 3), dtype=np.float32)
P[:, 1] = 1.0
if (~empty).any():
    idx = np.where(~empty)[0]
    P[idx] = score_batch([texts[i] for i in idx])

out = pd.DataFrame({
    "Datetime": df["Datetime"],
    "valence": P[:, 0] - P[:, 2],
    "p_pos": P[:, 0], "p_neu": P[:, 1], "p_neg": P[:, 2],
    "rescue": df["rescue"],
})
out.to_csv(OUT, index=False, compression="gzip")
el = time.time() - t0
print(f"BITTI: {len(df):,} tweet | {el:.0f}s ({len(df)/el:.0f}/sn) -> {OUT}", flush=True)
print(f"Valans: ort={out['valence'].mean():+.4f} std={out['valence'].std():.4f}", flush=True)
