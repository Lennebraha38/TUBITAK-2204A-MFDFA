"""
MF-DFA Cekirdek Modulu + Surrogate Ureticileri
TUBITAK 2204-A - Faruk Hilmi AKPINAR

Icerik:
  - pmodel_cascade          : binomial cascade sentetik seri
  - theoretical_alpha       : p-model analitik Holder ussu
  - mfdfa                   : coklu fraktal DFA (Kantelhardt 2002)
  - spectrum_width          : Delta_alpha hesaplama
  - surrogates              : IAAFT, shuffle, phase-randomized
  - surrogate_test          : gozlemlenen genislik vs surrogate null dagilimi
"""
import numpy as np


# ============================================================
# 1. P-MODEL (dogrulama icin)
# ============================================================

def pmodel_cascade(N, p):
    """Deterministik binomial multifractal cascade (N = 2^k)."""
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError("N = 2^k olmali")
    x = np.array([1.0])
    for _ in range(int(np.log2(N))):
        left, right = x * p, x * (1 - p)
        new = np.empty(2 * len(x))
        new[0::2] = left
        new[1::2] = right
        x = new
    return x / np.sum(x) * N


def theoretical_alpha(q, p):
    """p-model analitik Holder ussu alpha(q)."""
    pq, qmq = p ** np.asarray(q), (1 - p) ** np.asarray(q)
    return -(pq * np.log(p) + qmq * np.log(1 - p)) / ((pq + qmq) * np.log(2))


# ============================================================
# 1b. REFERANS SURECLER (dogrulama / kalibrasyon icin)
# ============================================================

def fgn(N, H, rng):
    """
    Fraksiyonel Gauss gurultu (fGn) - spektral sentez.
    Guc spektrumu |f|^{-(2H-1)}.

    Bu, projenin Hurst referansidir: DFA ile olculen uss H'dir
    (tests/test_mfdfa.py: H=0.3/0.5/0.7/0.9 -> 0.30/0.51/0.71/0.92).
    Tekli fraktaldir (Delta_alpha ~ 0), yani IAAFT null'unu ASMAMALI.
    """
    fr = np.fft.rfftfreq(N)
    fr[0] = 1e-12
    amp = fr ** (-(H - 0.5))
    ph = rng.uniform(0, 2 * np.pi, len(amp))
    return np.fft.irfft(amp * np.exp(1j * ph), n=N)


def mrw(N, sigma=0.5, rng=None, win=64):
    """
    Multifraktal Rastgele Yuruyus (MRW; Bacry, Delour & Muzy 2001).

        y_{t+1} = y_t + b_t * eps_t ,   b_t = exp(omega_t - sigma^2/2)

    omega_t: Gausssurec, uzun erimli korelasyonlu (PSD ~ 1/f^2).
    b_t: zamanla YAVAS DEGISTEN oynaklik (intermittency) -> MF-DFA bunu
    coklu fraktallik olarak gorur.

    MARGIN DAGILIMI DUZGUN (suretik) oldugu icin IAAFT surrogu bunu
    YIKAMAZ -> IAAFT testi bu surecte gecerlidir (p-modelin aksine).

    NOT: sigma arttikca Delta_alpha monoton artar (0.3->0.14, 0.5->0.55,
    0.8->1.00). sigma>0.8'de bant sinirlamasi artefaktlari belirir; bu
    yuzden kalibrasyonda sigma <= 0.8 kullanilir.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    fr = np.fft.rfftfreq(N)
    amp = np.zeros_like(fr)
    amp[1:] = 1.0 / fr[1:]           # PSD ~ 1/f^2
    ph = rng.uniform(0, 2 * np.pi, len(amp))
    om = np.fft.irfft(amp * np.exp(1j * ph), n=N)
    sd = om.std()
    if sd > 0:
        om = om / sd * np.sqrt(2.0) * sigma
    b = np.exp(om - 0.5 * sigma ** 2)
    eps = rng.standard_normal(N)
    return np.cumsum(b * eps)


def ar1(N, phi, rng):
    """Otokorelasyonlu AR(1) sureci: x_t = phi*x_{t-1} + e_t (dogrusal bellek)."""
    e = rng.standard_normal(N)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = phi * x[i - 1] + e[i]
    return x


def white_noise(N, rng):
    """Beyaz gurultu (rastgele, belleksiz)."""
    return rng.standard_normal(N)


# ============================================================
# 2. MF-DFA
# ============================================================

def mfdfa(x, q_vals, order=2, n_scales=30, s_min=32, s_max_frac=4):
    """
    Coklu Fraktal DFA.
    Donus: dict(h_q, tau_q, alpha, f_alpha, scales, Fq)

    Not: s_min=32. Gercek duygu serisinde s < 24'te capraz-gecis (kutu-ici
    ornekleme gurultusu) oldugu icin (R^2 0.63 -> 0.94) kucuk olcekler
    dislanir. p-model gibi temiz multifraktallarda Delta_alpha s_min'den
    bagimsizdir (~1.19).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    Y = np.cumsum(x - np.mean(x))

    s_min = max(s_min, order + 2)
    s_max = N // s_max_frac
    scales = np.unique(np.round(
        np.logspace(np.log10(s_min), np.log10(s_max), n_scales)
    ).astype(int))

    Fq = np.zeros((len(q_vals), len(scales)))
    for j, s in enumerate(scales):
        step = max(1, s // 2)
        starts = np.arange(0, N - s + 1, step)
        F2 = np.empty(len(starts))
        t = np.arange(s)
        for k, st in enumerate(starts):
            seg = Y[st:st + s]
            coef = np.polyfit(t, seg, order)
            res = seg - np.polyval(coef, t)
            F2[k] = np.mean(res ** 2)
        F2 = F2[F2 > 0]
        if len(F2) == 0:
            Fq[:, j] = np.nan
            continue
        for i, q in enumerate(q_vals):
            if abs(q) < 1e-10:
                # Standart MF-DFA: F_0(s) = exp( 1/2 * mean( ln F^2 ) )
                # (F(s,v)=sqrt(F2) oldugundan ln F = 1/2 ln F2)
                Fq[i, j] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                Fq[i, j] = np.mean(F2 ** (q / 2.0)) ** (1.0 / q)

    log_s = np.log(scales)
    h_q = np.full(len(q_vals), np.nan)
    for i in range(len(q_vals)):
        lf = np.log(Fq[i, :])
        m = np.isfinite(lf)
        if m.sum() >= 3:
            h_q[i] = np.polyfit(log_s[m], lf[m], 1)[0]

    tau_q = q_vals * h_q - 1.0
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    return dict(h_q=h_q, tau_q=tau_q, alpha=alpha, f_alpha=f_alpha,
                scales=scales, Fq=Fq)


def spectrum_width(res):
    """Delta_alpha = alpha_max - alpha_min (çoklu fraktallik genisligi)."""
    a = res["alpha"]
    a = a[np.isfinite(a)]
    return float(np.max(a) - np.min(a)) if len(a) else np.nan


def h_spread(res):
    """Delta_h = h(q_min) - h(q_max)."""
    h = res["h_q"]
    h = h[np.isfinite(h)]
    return float(h[0] - h[-1]) if len(h) else np.nan


# ============================================================
# 3. SURROGATE URETICILERI
# ============================================================

def shuffle_surrogate(x, rng):
    """Tam karistirma: hem korelasyon hem dagilim korunur, sira bozulur."""
    return rng.permutation(np.asarray(x, dtype=float))


def phase_surrogate(x, rng):
    """Faz rastgelelestirme: guc spektrumu korunur, dogrusal korelasyon korunur,
       fazlilik (nonlinearite) yok edilir. Gaussian seri uretir."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x - x.mean())
    amp = np.abs(X)
    phases = rng.uniform(0, 2 * np.pi, len(amp))
    phases[0] = 0.0
    if N % 2 == 0:
        phases[-1] = 0.0
    Xr = amp * np.exp(1j * phases)
    y = np.fft.irfft(Xr, n=N)
    return y


def iaaft_surrogate(x, rng, n_iter=100):
    """
    IAAFT (Schreiber & Schmitz 1996): hem guc spektrumu hem de
    genlik dagilimi korunur. MF-DFA'da standart null model.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    x_sorted = np.sort(x)
    X = np.fft.rfft(x)
    target_amp = np.abs(X)

    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        amp = np.abs(Y)
        amp[amp == 0] = 1e-12
        Y_new = target_amp * (Y / amp)
        y = np.fft.irfft(Y_new, n=N)
        # Genlik dagilimini geri yukle (siralama ile)
        ranks = np.argsort(np.argsort(y))
        y = x_sorted[ranks]
    return y


def surrogate_test(x, q_vals, n_surr=200, kind="iaaft", order=2,
                   n_scales=30, seed=42, stat="width"):
    """
    Gozlemlenen MF-DFA istatistigini surrogate null dagilimiyla karsilastir.
    stat: 'width' (Delta_alpha) veya 'hspread' (Delta_h)
    Donus: dict(observed, null_mean, null_std, z, p_value, null_array)
    """
    rng = np.random.default_rng(seed)
    res0 = mfdfa(x, q_vals, order=order, n_scales=n_scales)
    obs = spectrum_width(res0) if stat == "width" else h_spread(res0)

    null = np.empty(n_surr)
    for i in range(n_surr):
        if kind == "iaaft":
            s = iaaft_surrogate(x, rng)
        elif kind == "shuffle":
            s = shuffle_surrogate(x, rng)
        elif kind == "phase":
            s = phase_surrogate(x, rng)
        else:
            raise ValueError(kind)
        r = mfdfa(s, q_vals, order=order, n_scales=n_scales)
        null[i] = spectrum_width(r) if stat == "width" else h_spread(r)

    mu, sd = float(np.mean(null)), float(np.std(null))
    z = (obs - mu) / sd if sd > 0 else np.nan
    # p-deger (tek yonlu: gozlem > null)
    p = (np.sum(null >= obs) + 1) / (n_surr + 1)
    return dict(observed=obs, null_mean=mu, null_std=sd, z=z,
                p_value=float(p), null_array=null, result_obs=res0)
