"""
Cok boyutlu duygu skorlamasi (6 duygu).
Model: furkankarakuz/turkish-emotion-classifier
Kullanim: python score_emotions.py <input_csv> <text_col> <date_col> <output_csv>
"""
import sys
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "furkankarakuz/turkish-emotion-classifier"
DEVICE = "cuda"

in_csv, text_col, date_col, out_csv = sys.argv[1:5]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)
model.eval()
model.half()
id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
print(f"Model hazir. id2label={id2label}", flush=True)
emotions = [id2label[i].lower() for i in range(len(id2label))]

url_re = re.compile(r"http\S+|www\.\S+")
mention_re = re.compile(r"@\w+")
space_re = re.compile(r"\s+")


def clean(t):
    t = str(t)
    t = url_re.sub(" ", t)
    t = mention_re.sub(" ", t)
    return space_re.sub(" ", t.replace("RT ", " ")).strip()


@torch.inference_mode()
def score_batch(texts, bs=512):
    n = len(texts)
    nlab = len(id2label)
    out = np.zeros((n, nlab), dtype=np.float32)
    for i in range(0, n, bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(DEVICE)
        probs = F.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
        out[i:i + len(chunk)] = probs
    return out


t0 = time.time()
print(f"Okunuyor: {in_csv}", flush=True)
df = pd.read_csv(in_csv, usecols=[date_col, text_col])
df["clean"] = df[text_col].map(clean)
empty = (df["clean"].str.len() == 0).values
texts = df["clean"].tolist()

nlab = len(id2label)
P = np.zeros((len(df), nlab), dtype=np.float32)
P[:, 0] = 1.0
if (~empty).any():
    idx = np.where(~empty)[0]
    P[idx] = score_batch([texts[i] for i in idx])

out = pd.DataFrame({date_col: df[date_col]})
for j, e in enumerate(emotions):
    out[f"emo_{e}"] = P[:, j]
out.to_csv(out_csv, index=False, compression="gzip")
el = time.time() - t0
print(f"BITTI: {len(df):,} | {el:.0f}s -> {out_csv}", flush=True)
for j, e in enumerate(emotions):
    print(f"  {e:10s} ort={P[:,j].mean():.4f}", flush=True)
