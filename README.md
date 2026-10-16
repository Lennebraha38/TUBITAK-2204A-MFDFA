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

### 1. Duygu ölçümü — **model doğrulaması kritik**
Sosyal medya duygu ölçümünde **ölçüm geçerliliği** en kritik adımdır. Genel amaçlı
duygu modelleri felaket bağlamında sistematik hata yapabilir. Bu projede üç model
gerçek deprem tweet'lerinde test edildi:

| Model | Gerçek veride davranış | Karar |
|-------|------------------------|-------|
| `saribasmetehan/...sentiment-analysis` (3 sınıf) | "Acil yardım edin", "geçmiş olsun" → **POZİTİF** ❌ | elendi |
| `savasy/...sentiment-cased` (2 sınıf) | %70 negatif, makul ama kusurlu | **robustluk kontrolü** |
| `furkankarakuz/turkish-emotion-classifier` (6 duygu) | korku/üzüntü/öfke/sevinç örnekleri **tutarli** ✅ | **ANA ölçüm** |

- **ANA sinyaller (6 duygu):** anger, disgust, fear, joy, sadness, surprise
- **Bileşik:** `distress = sadness + fear`, `neg_emo = sad+fear+anger+disgust`,
  `valence_emo = joy − neg_emo`
- **Robustluk:** `val_sav = P(poz) − P(neg)` (savasy, bağımsız model)
- 1 dakikalık kutularda ortalama → $N \approx 7200$

> Bu doğrulama, "naif model seçimi yanlış bilimsel sonuç üretir" dersini somutlar;
> projenin metodolojik katkılarından biridir.

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
mfdfa_core.py         MF-DFA motoru + surrogate üreticileri (IAAFT/shuffle/faz)
pmodel_test_v2.py     P-model (binomial cascade) doğrulaması
calibration.py        MF-DFA kalibrasyon cetveli (bilinen süreçler)
verify_fix.py         h(q) ve Δα doğrulaması (fGn, p-model)
score2.py             ANA skorlama: duygu modeli (6 boyut) + savasy [both events]
compare_on_real.py    Model karşılaştırması (gerçek tweet'lerde)
emotion_inspect.py    Duygu modeli örnek doğrulaması
score_all.py          (eski) tek-model skorlama — arşiv
score_izmir.py        (eski) İzmir skorlama — arşiv
build_series2.py      1-dk/5-dk zaman serisi üretimi (tüm sinyaller)
analyze2.py           MF-DFA + surrogate testleri (seçilen sinyal)
run_analysis.py       Tüm analiz sürücüsü (iki olay × yedi sinyal)
compare_events2.py    Karşılaştırma tablosu + figür
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

# 3. Tweet'leri skorla (duygu modeli + savasy, iki olay)
python score2.py km2023
python score2.py izmir2020

# 4. Kalibrasyon + doğrulama (opsiyonel)
python calibration.py
python verify_fix.py

# 5. Tüm analiz: seri üretimi + her sinyalde MF-DFA + surrogate
python run_analysis.py 100

# 6. İki olayı karşılaştır
python compare_events2.py
```

---

## ✅ Doğrulama

- **Hurst üssü:** spektral sentezle üretilen fGn serilerinde $h(q{=}2)$,
  $H=0.3/0.5/0.7/0.9$ için $0.323/0.518/0.704/0.893$ → teorik değerlerle **uyumlu** ✓
- **P-model (binomial cascade):** $\Delta\alpha$ teorik $1.188$ (p=0.3),
  hesaplanan $1.202$ → **bağıl hata %1.2** ✓ (p=0.2: 1.996 → 2.009)
- **Kalibrasyon cetveli** (`calibration.py`): beyaz gürültü $0.08$, AR(1) $0.14$–$0.30$,
  fBm $0.21$–$0.43$ (monofraktal) vs. p-model $1.2$–$2.0$ (multifraktal)
- **MF-DFA çekirdek düzeltmesi:** $q=0$ dalı $F_0=\exp(\tfrac12\overline{\ln F^2})$
  olacak şekilde düzeltildi (önce $h(0)$ iki katı çıkıyordu; $\Delta\alpha$ etkilenmemişti)
- **Duygu modeli etiket eşlemesi:** model kartı + gerçek tweet örnekleriyle teyit ✓
- **Model seçimi:** 3 model gerçek tweet'lerde karşılaştırıldı, ölçüm hatası elendi ✓
- **Surrogate testleri:** bilinen süreçlerde doğrulandı ✓

---

## 📜 Lisans / Atıf

TÜBİTAK 2204-A proje çalışması. Yöntem: Kantelhardt vd. (2002),
Schreiber & Schmitz (1996) IAAFT.
