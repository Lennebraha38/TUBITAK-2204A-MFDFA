"""Hiz karsilastirma testi - FP16 ve batch boyutu."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "saribasmetehan/bert-base-turkish-sentiment-analysis"
df = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv",
                 usecols=["Tweet"], nrows=4000)
texts = df["Tweet"].astype(str).tolist()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cuda")
model.eval()
print("Hazir.", flush=True)


def run(tag, bs, maxlen, half):
    m = model.half() if half else model.float()
    n = len(texts)
    # isinma
    with torch.inference_mode():
        e = tok(texts[:16], padding=True, truncation=True, max_length=maxlen,
                return_tensors="pt").to("cuda")
        m(**e)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, n, bs):
            chunk = texts[i:i + bs]
            e = tok(chunk, padding=True, truncation=True, max_length=maxlen,
                    return_tensors="pt").to("cuda")
            lg = m(**e).logits
            _ = F.softmax(lg.float(), dim=-1).cpu().numpy()
    torch.cuda.synchronize()
    el = time.time() - t0
    print(f"{tag:35s}: {n/el:7.0f} tweet/sn | 942K icin {942180/(n/el)/60:5.1f} dk",
          flush=True)


run("A) fp32 bs128 len96", 128, 96, False)
run("B) fp16 bs256 len96", 256, 96, True)
run("C) fp16 bs512 len64", 512, 64, True)
run("D) fp16 bs512 len96", 512, 96, True)
run("E) fp16 bs1024 len64", 1024, 64, True)
