"""
MF-DFA cekirdek motoru icin otomatik dogrulama testleri.

Amac: projenin MATEMATIK dogrulugunu kanitlamak. Her test, motorun
bilinen-dogru bir referans surece verilen analitik sonucu vermesini
kontrol eder. Boylece "Delta_alpha buyuk" gozlemi, yazilim hatasindan
kaynaklanmis olamaz.

Calistirma:
    python -m pytest tests/ -v
"""
import numpy as np
import pytest

from mfdfa_core import (
    mfdfa, spectrum_width, h_spread, pmodel_cascade, theoretical_alpha,
    fgn, ar1, white_noise, mrw,
    shuffle_surrogate, phase_surrogate, iaaft_surrogate, surrogate_test,
)

Q = np.linspace(-5, 5, 21)
I2 = int(np.where(np.isclose(Q, 2.0))[0][0])
N_TEST = 4096


@pytest.fixture
def rng():
    return np.random.default_rng(20240206)


# ---------------------------------------------------------------
# 1. P-MODEL (Coklu fraktal referans)
# ---------------------------------------------------------------

def test_pmodel_requires_power_of_two():
    """N = 2^k degilse hata verilmeli."""
    with pytest.raises(ValueError):
        pmodel_cascade(1000, 0.3)


def test_pmodel_theory_width_matches_measured():
    """
    En kritik test: p-model icin teorik Delta_alpha ile hesaplanan
    degerin bagil hata %5'ten kucuk olmali.
    p=0.3 icin teorik ~1.19, p=0.2 icin ~2.0.
    """
    for p, theo in ((0.3, None), (0.2, None)):
        x = pmodel_cascade(4096, p)
        r = mfdfa(x, Q)
        meas = spectrum_width(r)
        theo = float(theoretical_alpha(Q, p).max() - theoretical_alpha(Q, p).min())
        rel_err = abs(meas - theo) / theo
        assert rel_err < 0.05, (
            f"p={p}: olculen {meas:.4f} != teorik {theo:.4f} "
            f"(bagil hata %{rel_err*100:.2f})")


def test_pmodel_width_grows_with_skew():
    """Daha asimetrik kaskad (kucuk p) -> daha genis spektrum."""
    widths = [spectrum_width(mfdfa(pmodel_cascade(4096, p), Q))
              for p in (0.5, 0.4, 0.3, 0.2)]
    assert widths == sorted(widths), f"genislikler monoton olmali: {widths}"


# ---------------------------------------------------------------
# 2. HURST USSU KURTARIMI
# ---------------------------------------------------------------

@pytest.mark.parametrize("H", [0.3, 0.5, 0.7, 0.9])
def test_fgn_recovers_hurst(H, rng):
    """fGn: DFA ile olculecek h(q=2) gercek H'ye yakin olmali."""
    x = fgn(N_TEST, H, rng)
    h2 = mfdfa(x, Q)["h_q"][I2]
    assert abs(h2 - H) < 0.10, f"H={H} icin h(q=2)={h2:.3f} (hata {abs(h2-H):.3f})"


def test_white_noise_hurst_is_half(rng):
    """Beyaz gurultu: h(q=2) = 0.5 (Hurst rastgele yuruyus)."""
    h2 = mfdfa(white_noise(N_TEST, rng), Q)["h_q"][I2]
    assert abs(h2 - 0.5) < 0.10, f"h(q=2)={h2:.3f} (0.5 olmali)"


def test_ar1_is_more_persistent_than_white_noise(rng):
    """Guclu otokorelasyon -> daha yuksek gorulen Hurst ussu."""
    h_white = mfdfa(white_noise(N_TEST, rng), Q)["h_q"][I2]
    h_ar = mfdfa(ar1(N_TEST, 0.9, rng), Q)["h_q"][I2]
    assert h_ar > h_white + 0.1, f"AR(1) h={h_ar:.3f} > beyaz h={h_white:.3f} olmali"


# ---------------------------------------------------------------
# 3. TEKLI / COKLU FRAKTALLIK AYRIMI
# ---------------------------------------------------------------

def test_white_noise_is_monofractal(rng):
    """Beyaz gurultu tekli fraktaldir: Delta_alpha kucuk olmali."""
    da = spectrum_width(mfdfa(white_noise(N_TEST, rng), Q))
    assert da < 0.25, f"beyaz gurultu Delta_alpha={da:.3f} (<0.25 olmali)"


def test_fgn_is_monofractal(rng):
    """fGn tekli fraktaldir: Delta_alpha kucuk, h(q) neredeyse duz olmali."""
    r = mfdfa(fgn(N_TEST, 0.7, rng), Q)
    assert spectrum_width(r) < 0.35, f"fGn Delta_alpha={spectrum_width(r):.3f}"
    assert h_spread(r) < 0.30, f"fGn Delta_h={h_spread(r):.3f} (duz olmali)"


def test_pmodel_is_multifractal(rng):
    """p-model coklu fraktaldir: Delta_alpha belirgin bicimde buyuk olmali."""
    da = spectrum_width(mfdfa(pmodel_cascade(4096, 0.3), Q))
    assert da > 0.8, f"p-model Delta_alpha={da:.3f} (>0.8 olmali)"


def test_multifractal_wider_than_monofractal():
    """Ayni uzunlukta: p-model fGn'den daha genis spektrum vermeli."""
    rng = np.random.default_rng(7)
    da_mono = spectrum_width(mfdfa(fgn(N_TEST, 0.6, rng), Q))
    da_multi = spectrum_width(mfdfa(pmodel_cascade(N_TEST, 0.3), Q))
    assert da_multi > 2 * da_mono, f"coklu {da_multi:.3f} vs tekli {da_mono:.3f}"


# ---------------------------------------------------------------
# 4. LEGENDRE DONUŞUMU TUTARLILIGI
# ---------------------------------------------------------------

def test_legendre_transform_consistent():
    """alpha = d(tau)/dq  ve  f = q*alpha - tau  birbirini tutarliyor olmali."""
    x = pmodel_cascade(4096, 0.3)
    r = mfdfa(x, Q)
    tau = r["tau_q"]
    alpha_num = np.gradient(tau, Q)
    assert np.allclose(alpha_num, r["alpha"], atol=1e-8)
    f_calc = Q * r["alpha"] - tau
    assert np.allclose(f_calc, r["f_alpha"], atol=1e-8)


def test_f_alpha_max_near_q_zero():
    """Spektrum tepe noktasi genellikle q=0 yakinindadir (f_max ~ 1)."""
    r = mfdfa(pmodel_cascade(4096, 0.3), Q)
    assert r["f_alpha"].max() > 0.7


# ---------------------------------------------------------------
# 5. SURROGATE URETICILERININ KORUMA/LIKEDE ETMEZLERI
# ---------------------------------------------------------------

def test_shuffle_preserves_distribution_destroys_order(rng):
    """Shuffle: dagilim korunur, otokorelasyon yok edilir."""
    x = ar1(N_TEST, 0.9, rng)
    y = shuffle_surrogate(x, rng)
    assert np.allclose(np.sort(x), np.sort(y)), "sirali degerler degismemeli"
    ac_x = np.corrcoef(x[:-1], x[1:])[0, 1]
    ac_y = np.corrcoef(y[:-1], y[1:])[0, 1]
    assert ac_x > 0.5, f"orijinal AC={ac_x:.2f} pozitif olmali"
    assert abs(ac_y) < 0.05, f"shuffle AC={ac_y:.3f} ~0 olmali"


def test_iaaft_preserves_distribution_and_spectrum(rng):
    """IAAFT: hem genlik dagilimi hem guc spektrumu korunur."""
    x = ar1(N_TEST, 0.9, rng)
    y = iaaft_surrogate(x, rng, n_iter=50)
    # (a) genlik dagilimi
    assert np.allclose(np.sort(x), np.sort(y), atol=1e-8), "genlik dagilimi bozulmus"
    # (b) guc spektrumu
    px = np.abs(np.fft.rfft(x - x.mean())) ** 2
    py = np.abs(np.fft.rfft(y - y.mean())) ** 2
    rel = np.abs(py - px).sum() / px.sum()
    assert rel < 0.10, f"guc spektrumu korunmamis (toplam hata %{rel*100:.1f})"


def test_phase_surrogate_preserves_spectrum(rng):
    """Faz-rastgele: guc spektrumu korunur."""
    x = ar1(N_TEST, 0.9, rng)
    y = phase_surrogate(x, rng)
    px = np.abs(np.fft.rfft(x - x.mean())) ** 2
    py = np.abs(np.fft.rfft(y - y.mean())) ** 2
    rel = np.abs(py - px).sum() / px.sum()
    assert rel < 1e-6, f"guc spektrumu korunmamis ({rel})"


# ---------------------------------------------------------------
# 6. SURROGATE TESTININ GECERLILIGI (yanlis pozitif riski yok mu?)
# ---------------------------------------------------------------

def test_mrw_multifractality_grows_with_sigma():
    """
    MRW (stokastik coklu fraktal): sigma arttikca Delta_alpha artar.
    (Kalibrasyon cetvelinin temel ozelligi; sabit tohumla deterministik.)
    """
    das = []
    for sg in (0.3, 0.5, 0.8):
        rng = np.random.default_rng(0)
        das.append(spectrum_width(mfdfa(mrw(4096, sigma=sg, rng=rng), Q)))
    assert das[0] < das[1] < das[2], f"MRW Delta_alpha monoton olmali: {das}"


def test_recorded_results_are_significant():
    """
    DEPO BUTUNLUK TESTI: kayitli ana sonuclar, raporda raporlanan bulgularla
    uyumlu olmali. Ana sinyal (distress) her iki olayda TUM penceresinde
    hem IAAFT hem shuffle null'unu anlamli bicimde asmali.
    """
    import os
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ev in ("km2023", "izmir2020"):
        f = os.path.join(root, "results", f"{ev}_distress_mfdfa.csv")
        if not os.path.exists(f):
            pytest.skip(f"sonuc dosyasi yok: {f}")
        df = pd.read_csv(f)
        tum = df[df.window == "TUM"].iloc[0]
        assert tum.iaaft_z > 5, f"{ev} TUM IAAFT z={tum.iaaft_z} (>5 olmali)"
        assert tum.iaaft_p < 0.01, f"{ev} TUM IAAFT p={tum.iaaft_p}"
        assert tum.shuffle_z > 5, f"{ev} TUM shuffle z={tum.shuffle_z} (>5 olmali)"
        assert tum.shuffle_p < 0.01, f"{ev} TUM shuffle p={tum.shuffle_p}"


def test_iaaft_limitation_on_discrete_cascade():
    """
    BELGELENMIS SINIRLAMA (bu bir hata degil, yontemin gercek davranisidir):
    p-model kaskadinin genlik dagilimi AYRIK oldugu icin IAAFT surrogu
    onun Delta_alpha'sini neredeyse tamamen taklit eder (z kucuk).
    Buna karsilik SHUFFLE ve FAZ-RASGELE null'lari anlamli sapma verir.

    Ders: IAAFT tek basina kullanilmamali; en az bir shuffle/faz kontrolu
    sart. Bu, raporda "Sinirliliklar" bolumunde belgelenmistir.
    """
    x = pmodel_cascade(4096, 0.3)
    st_i = surrogate_test(x, Q, n_surr=30, kind="iaaft", seed=3)
    st_sh = surrogate_test(x, Q, n_surr=30, kind="shuffle", seed=3)
    st_ph = surrogate_test(x, Q, n_surr=30, kind="phase", seed=3)
    assert st_i["z"] < 2.0, f"IAAFT beklenen sekilde zayif: z={st_i['z']:.2f}"
    assert st_sh["z"] > 2.0, f"shuffle yakalamali: z={st_sh['z']:.2f}"
    assert st_ph["z"] > 2.0, f"faz-r. yakalamali: z={st_ph['z']:.2f}"


def test_surrogate_test_null_for_white_noise():
    """
    YANLIS POZITIF KONTROLU: beyaz gurultu IAAFT null'unu ASMAMALI.
    (Bu test, atilmis bir z-skoruyla "yeni bulgu" sunmayi engeller.)
    """
    rng = np.random.default_rng(11)
    x = white_noise(N_TEST, rng)
    st = surrogate_test(x, Q, n_surr=40, kind="iaaft", seed=5)
    assert st["z"] < 2.0, f"beyaz gurultu icin z={st['z']:.2f} (anlamli olmamali)"
    assert st["p_value"] > 0.05, f"beyaz gurultu icin p={st['p_value']} (>0.05 olmali)"


def test_surrogate_test_null_for_monofractal_fgn():
    """fGn de anlamli sapma gostermemeli (tekli fraktal)."""
    rng = np.random.default_rng(13)
    x = fgn(N_TEST, 0.7, rng)
    st = surrogate_test(x, Q, n_surr=40, kind="iaaft", seed=9)
    assert st["z"] < 2.5, f"fGn icin z={st['z']:.2f} (anlamli olmamali)"


# ---------------------------------------------------------------
# 7. MF-DFA MOTORUNUN TEKNIK DOGRULUGU
# ---------------------------------------------------------------

def test_q_zero_branch_is_log_average():
    """
    q=0 dali F_0(s) = exp( mean(ln F^2) / 2 ) olmali.
    (Bu dalin iki kat hatali yazilmasi yaygin bir hatadir; burada korunuyor.)
    Kontrol: q=0 degeri, q=-0.5 ile q=+0.5 arasinda kalmali.
    """
    x = pmodel_cascade(4096, 0.3)
    res = mfdfa(x, Q)
    i0 = int(np.where(np.isclose(Q, 0.0))[0][0])
    i_p = int(np.where(np.isclose(Q, 0.5))[0][0])   # buyuk dalgalanmalar
    i_m = int(np.where(np.isclose(Q, -0.5))[0][0])  # kucuk dalgalanmalar
    f0 = res["Fq"][i0]
    assert np.all(f0 > 0), "F_0 pozitif olmali"
    assert np.all(f0 <= res["Fq"][i_p] * 1.001), "F_0 <= F_{q=0.5} olmali"
    assert np.all(f0 >= res["Fq"][i_m] * 0.999), "F_0 >= F_{q=-0.5} olmali"


def test_scales_are_sorted_and_unique(rng):
    res = mfdfa(white_noise(N_TEST, rng), Q)
    s = res["scales"]
    assert len(s) == len(set(s.tolist())), "olcekler tekrarli olmamali"
    assert np.all(np.diff(s) > 0), "olcekler sirali olmali"
    assert s[0] >= 32, "s_min=32 altina inilmemeli"


def test_result_shapes_consistent(rng):
    res = mfdfa(white_noise(N_TEST, rng), Q)
    assert len(res["h_q"]) == len(Q)
    assert len(res["alpha"]) == len(Q)
    assert res["Fq"].shape == (len(Q), len(res["scales"]))
    assert np.all(np.isfinite(res["h_q"])), "h_q sonlu olmali"


def test_spectrum_width_is_nonnegative(rng):
    for x in (white_noise(N_TEST, rng), pmodel_cascade(4096, 0.3)):
        assert spectrum_width(mfdfa(x, Q)) >= 0.0


def test_mfdfa_is_deterministic(rng):
    """Ayni girdi -> ayni sonuc (tohum bagimsiz: mfdfa deterministiktir)."""
    x = pmodel_cascade(4096, 0.3)
    a = spectrum_width(mfdfa(x, Q))
    b = spectrum_width(mfdfa(x, Q))
    assert a == b
