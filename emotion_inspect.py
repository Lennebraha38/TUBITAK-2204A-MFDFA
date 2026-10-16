"""Duygu modelinin GERCEK tweetlerde gecerliligini dogrula (ornek inceleme)."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda"
N = 4000
orig = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv", usecols=["Tweet"])
rng = np.random.default_rng(11)
idx = rng.choice(len(orig), size=N, replace=False)
texts = [str(orig["Tweet"].iloc[i]) for i in idx]

M = "furkankarakuz/turkish-emotion-classifier"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(DEVICE).eval().half()
labs = [str(v) for k, v in sorted(model.config.id2label.items())]

P = []
with torch.inference_mode():
    for i in range(0, N, 256):
        enc = tok(texts[i:i+256], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(DEVICE)
        P.append(F.softmax(model(**enc).logits.float(), -1).cpu().numpy())
P = np.vstack(P)

lines = []
for j, lab in enumerate(labs):
    order = np.argsort(P[:, j])[::-1][:8]
    lines.append("=" * 78)
    lines.append(f"EN YUKSEK: {lab}  (ort={P[:,j].mean():.3f}, argmax%={(P.argmax(1)==j).mean()*100:.1f})")
    lines.append("=" * 78)
    for i in order:
        lines.append(f"[{P[i,j]:.3f}] {texts[i][:115]}")
    lines.append("")

out = r"C:\TUBITAK_Proje\results\emotion_inspect.txt"
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("Yazildi:", out)
