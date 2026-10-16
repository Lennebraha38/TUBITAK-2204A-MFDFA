"""
BERTurk Turkce Duygu Modeli - Ilk Test
savasy/bert-base-turkish-sentiment-cased

Amac:
  1. Modelin GPU'da calistigini dogrulamak
  2. Deprem tweetlerinde duygu skoru dagilimini gormek
  3. Varyans var mi kontrol etmek (MF-DFA icin KRITIK)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "savasy/bert-base-turkish-sentiment-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("BERTURK DUYGU MODELI TESTI")
print("=" * 70)
print(f"Cihaz: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nModel indiriliyor...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)
model.eval()
print("Model yuklendi.")
print("Etiketler:", model.config.id2label)


def score_texts(texts, batch_size=128):
    """Her metin icin duygu skoru = P(pozitif) - P(negatif) in [-1, +1]"""
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1)
            # LABEL_0 = negatif, LABEL_1 = pozitif
            s = (probs[:, 1] - probs[:, 0]).cpu().numpy()
            scores.extend(s.tolist())
    return np.array(scores)


# ---------- 1. Sanity check ----------
print("\n" + "=" * 70)
print("1. ORNEK CUMLELER")
print("=" * 70)
examples = [
    "Deprem çok korkunçtu, herkes çok üzgün ve perişan",
    "Yardımlar bölgeye ulaştı, emeği geçen herkese teşekkürler",
    "Bugün hava çok güzel, parkta yürüyüş yaptım",
    "Enkaz altından bir kişi daha sağ olarak kurtarıldı",
    "Bu nasıl bir felaket, yüreğimiz yanıyor",
    "Allah yardımcımız olsun, dualarımız sizinle",
]
sc = score_texts(examples)
for t, s in zip(examples, sc):
    etiket = "POZITIF" if s > 0 else "NEGATIF"
    print(f"  [{s:+.3f}] {etiket:8s} | {t[:60]}")

# ---------- 2. Gercek deprem tweetleri ----------
print("\n" + "=" * 70)
print("2. GERCEK DEPREM TWEETLERI (N=5000 ornek)")
print("=" * 70)
df = pd.read_csv(r"C:\TUBITAK_Proje\data\earthquake\turkish\depremGun1tr.csv",
                 usecols=["Tweet"], nrows=50000)
texts = df["Tweet"].dropna().astype(str).tolist()
rng = np.random.default_rng(42)
idx = rng.choice(len(texts), size=min(5000, len(texts)), replace=False)
sample = [texts[i] for i in idx]

sc = score_texts(sample)

print(f"  Ortalama skor : {sc.mean():+.4f}")
print(f"  Std sapma     : {sc.std():.4f}")
print(f"  Min / Max     : {sc.min():+.4f} / {sc.max():+.4f}")
print(f"  Pozitif orani : {(sc > 0).mean() * 100:.1f}%")
print(f"  Negatif orani : {(sc < 0).mean() * 100:.1f}%")
print(f"  |skor|<0.5    : {(np.abs(sc) < 0.5).mean() * 100:.1f}% (belirsiz)")

# Dagilim histogrami (metin)
hist, edges = np.histogram(sc, bins=10, range=(-1, 1))
print("\n  Skor dagilimi:")
for i in range(len(hist)):
    bar = "#" * int(hist[i] / max(hist) * 40) if max(hist) > 0 else ""
    print(f"  [{edges[i]:+.1f},{edges[i+1]:+.1f}) {hist[i]:>5} {bar}")

print("\n" + "=" * 70)
if sc.std() > 0.05:
    print("SONUC: Varyans YETERLI -> MF-DFA icin uygun seri uretilebilir.")
else:
    print("SONUC: Varyans DUSUKE -> dikkat, skorlama stratejisi gozden gecirilmeli.")
print("=" * 70)
