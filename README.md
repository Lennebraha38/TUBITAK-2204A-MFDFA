# Sosyal Medya Duygu Zaman Serilerinin Çoklu Fraktal Analizi

**TÜBİTAK 2204-A — Matematik Alanı / Algoritma Tasarımı ve Uygulamaları**

> Deprem sonrası sosyal medya duygu zaman serilerinde **kaotik (fraktal)** ve
> **stokastik (rastgele)** dinamiklerin çoklu fraktal DFA (MF-DFA) ve surrogate
> analizi ile ayrıştırılması.

---

## 🎯 Amaç

Sosyal medyada bir kriz anında oluşan **kolektif duygu zaman serisinin**
istatistiksel doğası rastgele mi, yoksa çoklu fraktal (doğrusal olmayan,
uzun-erimli bağımlı) mı? Bu proje:

1. Türkçe tweet'lerden **sürekli bir duygu (valans) zaman serisi** üretir,
2. **MF-DFA** ile çoklu fraktallik genişliğini ($\Delta\alpha$) ölçer,
3. **IAAFT / shuffle / faz-rastgeleleştirme** surrogate testleriyle gözlemi
   bir "stokastik null model" ile karşılaştırır,
4. **Kriz (akut)** ile **rutin (normalleşme)** fazlarını karşılaştırır,
5. İki bağımsız deprem olayında **genellenebilirliği** test eder.

---

## 📊 Veri

| Olay | Tweet | Süre | Metin | Zaman damgası |
|------|-------|------|-------|----------------|
| Kahramanmaraş (6 Şub 2023) | ~942,000 | 5 gün | ✅ | ✅ |
| İzmir (30 Eki 2020) | ~971,000 | 8 gün | ✅ | ✅ |

> Veri dosyaları boyut nedeniyle bu repoda **yer almaz** (`data/` klasörü
> `.gitignore` ile hariç tutulmuştur). Kaynak: Kaggle.
> - `esatakkasoglu/turkey-06-02-2023-eartquake-tweets`
> - `aliburakcan/earthquake-turkish-tweets-partially-annotated` (İzmir)

---

## 🧮 Yöntem

### 1. Duygu ölçümü
- Model: `saribasmetehan/bert-base-turkish-sentiment-analysis` (BERTurk, 3 sınıf, %96.2)
- Etiketler: `LABEL_0 = nötr`, `LABEL_1 = pozitif`, `LABEL_2 = negatif`
- Sürekli skor: $S_t = P(\text{poz}) - P(\text{neg}) \in [-1, +1]$
- 1 dakikalık kutularda ortalama → $N \approx 7200$

### 2. Çoklu Fraktal DFA (Kantelhardt, 2002)
- Profil: $Y(i) = \sum_{k=1}^{i}(x_k - \bar{x})$
- Dalgalanma fonksiyonu $F_q(s)$, genelleştirilmiş Hurst üssü $h(q)$
- Legendre dönüşümü → $\tau(q)$, $\alpha = d\tau/dq$, $f(\alpha) = q\alpha - \tau$
- **Genişlik:** $\Delta\alpha = \alpha_{max} - \alpha_{min}$

### 3. Surrogate (null model) testleri
| Surrogate | Korur | Yok eder |
|-----------|-------|----------|
| **Shuffle** | Genlik dağılımı | Tüm zaman yapısı |
| **Faz-rastgele** | Güç spektrumu | Faz (doğrusal olmayan yapı) |
| **IAAFT** | Güç spektrumu + dağılım | Doğrusal olmayan çoklu fraktallik |

> Gerçek serinin $\Delta\alpha$'sı IAAFT null dağılımından **anlamlı biçimde
> büyükse** → seri gerçek çoklu fraktaldır (rastgele değil).

---

## 📁 Proje Yapısı

```
mfdfa_core.py         MF-DFA motoru + surrogate üreticileri
pmodel_test_v2.py     P-model (binomial cascade) doğrulaması  [Δα hata %4.4]
score_all.py          Kahramanmaraş 2023 tweet skorlama (fp16, GPU)
score_izmir.py        İzmir 2020 tweet skorlama
score_emotions.py     6 boyutlu duygu skorlama
build_series.py       1-dk / 5-dk zaman serisi üretimi
analyze_series.py     MF-DFA + surrogate testleri (tek olay)
compare_events.py     İki olay karşılaştırması
compare_models.py     Duygu modeli karşılaştırması
analyze_phases.py     Deprem verisi faz (hacim) analizi
izmir_analyze.py      İzmir verisi yapı analizi
test_*.py             Hız ve doğruluk testleri
```

---

## ▶️ Çalıştırma

```bash
# 1. Bağımlılıklar
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate numpy scipy pandas matplotlib seaborn numba

# 2. Veriyi indir (Kaggle) -> data/earthquake/, data/izmir/

# 3. Tweet'leri skorla
python score_all.py            # Kahramanmaraş 2023
python score_izmir.py          # İzmir 2020

# 4. Zaman serisi üret
python build_series.py "results\scored\deprem*_scored.csv.gz" Date km2023
python build_series.py "results\scored\izmir_scored.csv.gz" Datetime izmir2020

# 5. MF-DFA + surrogate
python analyze_series.py results\km2023_series_1min.csv km2023 2023-02-06T00:00
python analyze_series.py results\izmir2020_series_1min.csv izmir2020 2020-10-30T11:51

# 6. İki olayı karşılaştır
python compare_events.py
```

---

## ✅ Doğrulama

- **P-model (binomial cascade):** teorik $\Delta\alpha = 1.2224$ (p=0.3),
  hesaplanan $1.1689$ → **bağıl hata %4.37** ✓
- **Duygu modeli etiket eşlemesi:** model kartı + 12 cümlelik probe ile teyit ✓
- **Surrogate testleri:** bilinen süreçlerde doğrulandı ✓

---

## 📜 Lisans / Atıf

TÜBİTAK 2204-A proje çalışması. Yöntem: Kantelhardt vd. (2002),
Schreiber & Schmitz (1996) IAAFT.
