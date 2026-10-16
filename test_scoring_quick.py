"""Hizli test: model + skorlama 3000 tweet uzerinde, tamponsuz ciktilar."""
import sys, time, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("1) Baslatildi", flush=True)
print("   torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("   GPU:", torch.cuda.get_device_name(0), flush=True)

MODEL = "saribasmetehan/bert-base-turkish-sentiment-analysis"
print("2) Tokenizer yukleniyor...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
print("3) Model yukleniyor...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cuda")
model.eval()
print("4) Model hazir. id2label:", model.config.id2label, flush=True)

print("5) Veri okunuyor...", flush=True)
df = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv",
                 usecols=["Date", "Tweet"], nrows=3000)
print("6) DataFrame:", df.shape, flush=True)
texts = df["Tweet"].astype(str).tolist()

print("7) Skorlama basliyor...", flush=True)
t0 = time.time()
L_POS, L_NEU, L_NEG = 1, 0, 2
out = np.zeros((len(texts), 3), dtype=np.float32)
BS = 128
try:
    with torch.no_grad():
        for i in range(0, len(texts), BS):
            chunk = texts[i:i + BS]
            enc = tok(chunk, padding=True, truncation=True, max_length=96,
                      return_tensors="pt").to("cuda")
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).float().cpu().numpy()
            out[i:i + len(chunk), 0] = probs[:, L_POS]
            out[i:i + len(chunk), 1] = probs[:, L_NEU]
            out[i:i + len(chunk), 2] = probs[:, L_NEG]
            if i % (BS * 5) == 0:
                print(f"   ... {i}/{len(texts)} | {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print("!!! HATA:", type(e).__name__, e, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

el = time.time() - t0
val = out[:, 0] - out[:, 2]
print(f"8) BITTI. {len(texts)} tweet, {el:.1f}s = {len(texts)/el:.0f} tweet/sn", flush=True)
print(f"   Valans: ort={val.mean():+.3f} std={val.std():.3f}", flush=True)
print(f"   Tahmini 942K sure: {942180/(len(texts)/el)/60:.1f} dakika", flush=True)
