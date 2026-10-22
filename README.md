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
| Kahramanmaraş (6 Şub 2023) | 942,180 | 5 gün | ✅ | ✅ |
| İzmir (30 Eki 2020) | 971,295 | 8 gün | ✅ | ✅ |

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
- 1 dakikalık kutularda ortalama → $N \approx 7200$ (km2023) / $N \approx 10700$ (İzmir)

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

### 4. Sabit analiz protokolü
| Parametre | Değer |
|-----------|-------|
| $s_{min}$ | 32 (ölçek çapraz-geçişini önler) |
| $s_{max}$ | $N/4$ |
| $q$ | $-5 \dots +5$ (21 değer) |
| DFA mertebesi | 2 (polinom detrend) |
| Ölçek sayısı | 30 |
| Surrogate sayısı | 100 (her sinyal × pencere) |
| Pencereler | TÜM / AKUT (+24 s) / NORM (son 48 s) / HACİM (kontrol) |
| Analiz penceresi | $[t_{\text{olay}} - 1\,\text{s},\ t_{\text{son}}]$ |

---

## 📈 ANA SONUÇLAR

### Ana sinyal `distress` (= sadness + fear) — pencere karşılaştırması

| Olay | Pencere | $N$ | $\Delta\alpha$ | $\Delta h$ | IAAFT $z$ | IAAFT $p$ | Shuffle $z$ | Shuffle $p$ |
|------|---------|-----|-----------|-----------|-----------|-----------|-------------|-------------|
| km2023 | **TÜM** | 7192 | **0.779** | 0.502 | **+11.59** | 0.0099 | **+20.02** | 0.0099 |
| km2023 | AKUT | 1432 | 0.860 | 0.504 | −0.83 | 0.812 | +7.12 | 0.0099 |
| km2023 | NORM | 2881 | 0.378 | 0.221 | +7.22 | 0.0099 | +4.18 | 0.0099 |
| km2023 | HACİM | 7192 | 0.997 | 0.700 | +0.31 | 0.069 | +32.96 | 0.0099 |
| izmir2020 | **TÜM** | 10689 | **0.669** | 0.388 | **+10.19** | 0.0099 | **+21.55** | 0.0099 |
| izmir2020 | AKUT | 1440 | 0.745 | 0.449 | +6.75 | 0.0099 | +8.23 | 0.0099 |
| izmir2020 | NORM | 2881 | 0.246 | 0.075 | +2.72 | 0.0495 | +3.25 | 0.0099 |
| izmir2020 | HACİM | 10689 | 1.019 | 0.587 | +9.67 | 0.0099 | +40.37 | 0.0099 |

### Tüm sinyaller — TÜM penceresi

| Sinyal | km2023 $\Delta\alpha$ | km2023 IAAFT $z$ | izmir2020 $\Delta\alpha$ | izmir2020 IAAFT $z$ |
|--------|-----------|-----------|-----------|-----------|
| `distress` (ANA) | 0.779 | **+11.59** | 0.669 | **+10.19** |
| `anger` | 1.263 | **+22.64** | 0.666 | **+19.56** |
| `joy` | 0.873 | **+20.69** | 0.841 | **+10.40** |
| `valence_emo` | 0.861 | **+18.83** | 0.797 | **+20.53** |
| `val_sav` (robustluk) | 0.516 | **+10.43** | 0.428 | **+12.61** |
| `sadness` | 0.414 | **+7.10** | 0.289 | **+4.92** |
| `fear` | 0.572 | **+4.45** | 0.515 | **+10.54** |

### Yorum

1. **TÜM penceresinde tüm sinyaller, iki olayda da** IAAFT ve shuffle null'unu
   anlamlı biçimde aşar ($z \gtrsim 4.5$, $p \le 0.0099$). Duygu serileri
   **doğrusal olmayan çoklu fraktaldir** — rastgele değildir.
2. **Normalleşme (NORM)** penceresinde $\Delta\alpha$ belirgin biçimde düşer
   (0.78→0.38 ve 0.67→0.25): çoklu fraktallik kriz dönemine bağlıdır ve
   normale dönüşte sönümlenir.
3. **IAAFT null ortalaması düşüktür** (0.05–0.23): gözlenen çoklu fraktallik
   null modellerin üretebileceği bir artefakt değildir.
4. **Genellenebilirlik:** iki bağımsız olay, iki farklı model, iki farklı veri
   hacmi — ana bulgu her ikisinde de aynı yönde ve aynı büyüklük mertebesinde.

---

## 📁 Proje Yapısı

```
run_all.py            ★ TEK GİRİŞ NOKTASI (uçtan uca boru hattı)
mfdfa_core.py         MF-DFA motoru + surrogate üreticileri (IAAFT/shuffle/faz)
                      + referans süreçler (fGn, AR(1), beyaz gürültü, MRW)
calibration.py        Δα kalibrasyon cetveli -> results/calibration_table.csv
analyze2.py           MF-DFA + surrogate testleri (seçilen sinyal)
build_series2.py      1-dk/5-dk zaman serisi üretimi (tüm sinyaller)
score2.py             ANA skorlama: duygu modeli (6 boyut) + savasy [both events]
compare_events2.py    Karşılaştırma tablosu + figür
verify_fix.py         h(q) ve Δα doğrulaması (fGn, p-model)
run_analysis.py       Analiz sürücüsü (iki olay × yedi sinyal)
tests/                pytest paketi (28 test) — matematik doğrulama
rapor/proje_detay.tex Proje detay raporu (XeLaTeX)
rapor/proje_detay.txt Aynı raporun düz metin sürümü
requirements.txt      Sabitlenmiş bağımlılıklar
```

> Teşhis/keşif scriptleri (`robust_check.py`, `r2_check.py`, `izmir_windows.py`,
> `diurnal_check.py`, `compare_models*.py`, `qsensitivity.py`, `locslope.py`,
> `scale_test.py`, `pmodel_test_v2.py` vb.) de repoda durmaktadır; rapordaki
> sağlamlık kontrollerinin yeniden üretilebilmesi için gereklidir.

---

## ▶️ Çalıştırma

```bash
# 1. Bağımlılıklar
pip install -r requirements.txt
# (torch CUDA için: pip install torch --index-url https://download.pytorch.org/whl/cu124)

# 2. Veriyi indir (Kaggle) -> data/earthquake/, data/izmir/

# 3. ★ Tek komutla tüm boru hattı
python run_all.py
```

`run_all.py` sırasıyla: **veri denetimi → skorlama (GPU) → 1-dk seri üretimi →
MF-DFA + surrogate (14 çıktı) → kalibrasyon cetveli → karşılaştırma → pytest**.
Var olan çıktılar **atlanır** (yeniden başlatılabilir); `--force` ile hepsi
yeniden hesaplanır, `--stage N` ile tek aşama çalıştırılır.

Tek tek çalıştırmak isterseniz:

```bash
python score2.py km2023 && python score2.py izmir2020   # skorlama (GPU)
python build_series2.py "results/scored2/deprem*_score2.csv.gz" Date km2023
python analyze2.py results/km2023_series_1min.csv km2023 distress 2023-02-06T00:00 24 100
python calibration.py
python compare_events2.py
python -m pytest tests/ -q
```

---

## ✅ Doğrulama

### Matematik doğrulama (test paketi: 28 test, tamamı geçiyor)

- **Hurst üssü kurtarımı:** spektral sentezle üretilen fGn serilerinde
  $h(q{=}2)$, $H=0.3/0.5/0.7/0.9$ için $0.317/0.493/0.705/0.894$ → teorik
  değerlerle **uyumlu** ✓
- **P-model (binomial cascade):** $\Delta\alpha$ teorik $1.1875$ (p=0.3),
  hesaplanan $1.1639$ → **bağıl hata %2.0** ✓ (p=0.2: 1.996 → 1.952, %2.2)
- **$q=0$ dalı:** $F_0(s)=\exp(\tfrac12\overline{\ln F^2})$ olarak düzeltildi
  (önce $h(0)$ iki katı çıkıyordu; $\Delta\alpha$ etkilenmemişti) ✓
- **Surrogate testleri:** beyaz gürültü ve fGn'de anlamlı sapma yok ✓
- **Sonuç bütünlüğü:** depodaki kayıtlı sonuçların ana bulgularla uyumlu
  olduğu test edilir ✓

### Kalibrasyon cetveli ($\Delta\alpha$, $q\in[-5,5]$, $s_{min}=32$, $N{=}8192$)

| Süreç | Grup | $\Delta\alpha$ | $h(q{=}2)$ | Teorik | IAAFT $z$ |
|-------|------|-----------|-----------|--------|-----------|
| Beyaz gürültü | rastgele | 0.076 | 0.515 | — | +1.87 |
| Tekdüze dağılım | rastgele | 0.016 | 0.491 | — | — |
| Rastgele yürüyüş | rastgele | 0.095 | 1.439 | — | — |
| fGn $H=0.3$ | tekli fraktal | 0.017 | **0.317** | 0.30 ✓ | — |
| fGn $H=0.5$ | tekli fraktal | 0.077 | **0.493** | 0.50 ✓ | — |
| fGn $H=0.7$ | tekli fraktal | 0.025 | **0.705** | 0.70 ✓ | — |
| fGn $H=0.9$ | tekli fraktal | 0.044 | **0.894** | 0.90 ✓ | — |
| AR(1) $\phi=0.5$ | doğrusal bellek | 0.119 | 0.562 | — | — |
| AR(1) $\phi=0.9$ | doğrusal bellek | 0.333 | 0.770 | — | — |
| AR(1) $\phi=0.98$ | doğrusal bellek | 0.309 | 1.189 | — | — |
| p-model $p=0.4$ | çoklu fraktal (det.) | 0.514 | 0.889 | 0.449 | — |
| p-model $p=0.3$ | çoklu fraktal (det.) | 1.164 | 0.793 | **1.188** ✓ | **+0.99** ⚠ |
| p-model $p=0.2$ | çoklu fraktal (det.) | 1.952 | 0.656 | **1.996** ✓ | — |
| MRW $\sigma=0.3$ | çoklu fraktal (stok.) | 0.139 | 1.524 | — | — |
| MRW $\sigma=0.5$ | çoklu fraktal (stok.) | 0.547 | 1.426 | — | — |
| MRW $\sigma=0.8$ | çoklu fraktal (stok.) | 1.004 | 1.438 | — | **+3.20** |
| **Duygu serisi (distress)** | — | **0.67–0.78** | — | — | **+10…+12** |

> Gerçek verideki etki büyüklüğü ($z \approx +10\text{–}12$), **hiçbir referans
> sürecin** ulaşamadığı düzeydedir ($|z| \le 3.2$).

### Ölçek aralığı belirleme (kritik)
Gerçek duygu serisinde $s<24$ dk'da **çapraz-geçiş** vardır (kutu-içi örnekleme
gürültüsü): $s_{min}=16$ ile güç-yasası fiti $R^2=0.63$ (kötü), $s_{min}\ge 24$ ile
$R^2\approx 0.94$ (iyi). Temiz bir p-model referansında $R^2=0.99$ ve
$\Delta\alpha$ $s_{min}$'den bağımsızdır ($\approx 1.19$). Bu nedenle protokol
**$s_{min}=32$**, $s_{max}=N/4$, $q\in[-5,5]$, 30 ölçek olarak sabitlendi.

### Trend robustluğu
Büyük ölçeklerdeki global trendin $\Delta\alpha$'yı şişirip şişirmediği test edildi:
lineer/kuadratik/kübik detrend sonrası sonuç **korunur** (bkz. `robust_check.py`,
`scale_test.py`). Ayrıca 1. fark ve günlük-ortalama-çıkarma varyantları incelendi.

### Analiz penceresi kuralı (kritik)
Her olay için analiz penceresi $[\,t_{\text{olay}}-1\text{s},\ t_{\text{son}}\,]$
olarak tanımlandı. Gerekçe: İzmir verisinde olay-öncesi **gece**
(2020-10-29 21:07 → 2020-10-30 10:00) aşırı seyrektir (884 dakikanın 586'sı boş
kutu, %66) ve lineer interpolasyon büyük ölçeklerde **yapay** yapı üretir: ham
$\Delta\alpha=3.55 \to$ düzeltilmiş $\approx 0.67$ (`izmir_windows.py`,
`check_izmir_onset.py`). km2023'te kural etkisizdir (gece yeterince yoğun).
Pencereler: **TÜM**, **AKUT** ($t_{\text{olay}}+24$s), **NORM** (son 48s),
**HACİM** ($\log(1+\text{tweet/dk})$, kontrol).

### Sirkadiyen (24s) kontrolü
`diurnal_check.py` ile saatlik ortalama çıkarıldı; km2023'te $\Delta\alpha$
değişmez ($0.78\to0.86$), yani sonuç sirkadiyen döngü artefaktı değildir.

---

## ⚠️ Sınırlılıklar (rapora da işlenmiştir)

1. **IAAFT'nin ayrık genlik dağılımlı serilerdeki sınırlaması.**
   p-model kaskadının genlik dağılımı **ayrıktır** (58 farklı değer) ve IAAFT
   surrogu bu serinin $\Delta\alpha$'sını neredeyse tamamen taklit eder
   ($z=+0.99$, anlamsız). Buna karşılık **shuffle** ($z=+4.83$) ve **faz-rastgele**
   ($z=+20.05$) null'ları çoklu fraktalliği yakalar. Sonuç: **IAAFT tek başına
   yeterli bir null değildir**; her zaman en az bir shuffle/faz kontrolü
   birlikte raporlanmalıdır. Gerçek duygu serilerinin genlik dağılımı düzgün
   (sürekli) olduğu için ana bulgu bu sınırlamadan etkilenmez.
2. **Hacim (HACİM) kontrolü temiz bir null değildir.** $\log(1+\text{tweet/dk})$
   serisi de çoklu fraktaldir ($\Delta\alpha\approx 1.0$); IAAFT davranışı iki
   olay arasında tutarsızdır (km2023 $z=+0.31$, İzmir $z=+9.67$). Bunun nedeni
   km2023'teki 60 sıfır-dakikanın IAAFT null'unu kararsızlaştırmasıdır. Bu
   yüzden HACİM yalnızca **betimsel bir kontrol** olarak raporlanır; ana
   argüman duygu–hacim ayrımına değil, doğrudan **IAAFT + shuffle çift null
   testine** dayanır.
3. **AKUT penceresi kısıtılıdır.** $N\approx 1440$ ile ölçek aralığı yalnızca
   ~1 dekad olduğundan $\Delta\alpha$ ve $z$ gürültülüdür; AKUT penceresi
   **betimsel** olarak sunulur (km2023'te IAAFT $z=-0.83$, İzmir'de $+6.75$).
4. **Gözlemsel veri**: nedensellik iddiası yoktur; bulgular korelasyoneldir.
5. **Duygu modeli çıktısı bir ölçüm hatası kaynağıdır**; `saribasmetehan`
   modelinin elenmesi bu riskin somut kanıtıdır.

---

## 📄 Rapor

Proje detay dosyası (XeLaTeX, konuyu hiç bilmeyen biri için sıfırdan öğretici
anlatım; giriş → kuramsal arka plan → yöntem → bulgular → tartışma →
sonuç + ekler):

```
rapor/proje_detay.tex     (XeLaTeX)
rapor/proje_detay.txt     (düz metin sürümü)
```

Düz metin sürüm `convert_tex_txt.py` ile otomatik üretilir:

```bash
python convert_tex_txt.py
```

---

## 📜 Lisans / Atıf

TÜBİTAK 2204-A proje çalışması. Yöntem: Kantelhardt vd. (2002),
Schreiber & Schmitz (1996) IAAFT; Bacry, Delour & Muzy (2001) MRW.
