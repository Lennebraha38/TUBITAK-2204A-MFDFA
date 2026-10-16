"""
Ikinci nesil skorlama: DUYGU modeli (6 boyut) + savasy (valans robustluk kontrolu).

Neden: saribasmetehan 3-sinif modeli felaket baglamindaki "yardim/dua/teselli"
dilini yanlislikla POZITIF okuyor (olcum gecerliligi hatasi). Duygu modeli
gercek tweetlerde dogrulandi (korku/uzuntu/ofke/sevinc ornekleri tutarli).

Cikti kolonlari:
  Date, anger, disgust, fear, joy, sadness, surprise,   (duygu modeli)
  val_sav, p_sav_pos, p_sav_neg                        (savasy valans)

Kullanim:
  python score2.py km2023
  python score2.py izmir2020
"""
import argparse
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda"
EMO_MODEL = "furkankarakuz/turkish-emotion-classifier"
SAV_MODEL = "savasy/bert-base-turkish-sentiment-cased"
EMO_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

DATASETS = {
    "km2023": {
        "text_col": "Tweet",
        "date_col": "Date",
        "outdir": r"C:\TUBITAK_Proje\results\scored2",
        "files": [rf"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun{i}tr.csv"
                  for i in range(1, 6)],
        "tag": "km2023_day",
    },
    "izmir2020": {
        "text_col": "Text",
        "date_col": "Datetime",
        "outdir": r"C:\TUBITAK_Proje\results\scored2",
        "files": [r"C:\TUBITAK_Proje\data\izmir\izmir_earthquake.csv"],
        "tag": "izmir2020",
    },
}

url_re = re.compile(r"http\S+|www\.\S+")
mention_re = re.compile(r"@\w+")
space_re = re.compile(r"\s+")


def clean(t):
    t = str(t)
    t = url_re.sub(" ", t)
    t = mention_re.sub(" ", t)
    t = t.replace("RT ", " ")
    return space_re.sub(" ", t).strip()


print("Modeller yukleniyor...", flush=True)
emo_tok = AutoTokenizer.from_pretrained(EMO_MODEL)
emo = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL).to(DEVICE).eval().half()
sav_tok = AutoTokenizer.from_pretrained(SAV_MODEL)
sav = AutoModelForSequenceClassification.from_pretrained(SAV_MODEL).to(DEVICE).eval().half()
print(f"  duygu id2label : {emo.config.id2label}", flush=True)
print(f"  savasy id2label: {sav.config.id2label}", flush=True)


@torch.inference_mode()
def score_batch(texts, bs=512):
    n = len(texts)
    E = np.zeros((n, 6), dtype=np.float32)
    S = np.zeros((n, 2), dtype=np.float32)
    for i in range(0, n, bs):
        chunk = texts[i:i + bs]
        e = emo_tok(chunk, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(DEVICE)
        E[i:i + len(chunk)] = F.softmax(emo(**e).logits.float(), -1).cpu().numpy()
        s = sav_tok(chunk, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(DEVICE)
        S[i:i + len(chunk)] = F.softmax(sav(**s).logits.float(), -1).cpu().numpy()
    return E, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    os.makedirs(cfg["outdir"], exist_ok=True)

    t_all = time.time()
    for fi, fpath in enumerate(cfg["files"], 1):
        base = os.path.basename(fpath)
        out_path = os.path.join(cfg["outdir"], base.replace(".csv", "_score2.csv.gz"))
        if os.path.exists(out_path):
            print(f"ATLANDI (var): {out_path}", flush=True)
            continue
        t0 = time.time()
        df = pd.read_csv(fpath, usecols=[cfg["date_col"], cfg["text_col"]])
        df["clean"] = df[cfg["text_col"]].map(clean)
        empty = (df["clean"].str.len() == 0).values
        texts = df["clean"].tolist()

        E = np.zeros((len(df), 6), dtype=np.float32)
        S = np.zeros((len(df), 2), dtype=np.float32)

        if (~empty).any():
            idx = np.where(~empty)[0]
            E[idx], S[idx] = score_batch([texts[i] for i in idx])

        out = pd.DataFrame({cfg["date_col"]: df[cfg["date_col"]]})
        for j, name in enumerate(EMO_ORDER):
            out[name] = E[:, j]
        out["val_sav"] = S[:, 1] - S[:, 0]
        out["p_sav_pos"] = S[:, 1]
        out["p_sav_neg"] = S[:, 0]
        out.to_csv(out_path, index=False, compression="gzip")
        el = time.time() - t0
        print(f"  [{fi}/{len(cfg['files'])}] {base}: {len(df):,} tweet | "
              f"{el:.0f}s ({len(df)/el:.0f}/sn) -> {out_path}", flush=True)

    print(f"\nTOPLAM: {(time.time()-t_all)/60:.1f} dk", flush=True)
    print("BITTI", flush=True)


if __name__ == "__main__":
    main()
