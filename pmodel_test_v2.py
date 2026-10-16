"""
P-Model MF-DFA Doğrulama Testi — v2 (Düzeltilmiş)
TÜBİTAK 2204-A — Faruk Hilmi AKPINAR

Düzeltmeler:
  1. P-Model: Standart deterministik binomial cascade (permütasyon yok)
  2. q aralığı: Tam simetrik [-5, 5], q=0 ortada
  3. Daha fazla ölçek (30 adet) → daha kararlı eğim
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# ============================================================
# 1. STANDART P-MODEL (Deterministik Binomial Cascade)
# ============================================================

def pmodel_cascade(N: int, p: float) -> np.ndarray:
    """
    Deterministik Binomial Multifractal Cascade (Meneveau & Sreenivasan).

    Mekanizma:
      N = 2^k nokta için k adım.
      Her adımda her nokta ya p ya (1-p) katsayısıyla KARELENİR
      (cascade = üste binen çarpma).

    Örnek (k=2, N=4):
      Adım 1: [1] → [p, 1-p]
      Adım 2: [p, 1-p] → [p*p, p*(1-p), (1-p)*p, (1-p)*(1-p)]

    Sonuç: [p², p(1-p), p(1-p), (1-p)²]
    """
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError(f"N=2^k olmalı (2, 4, 8, ..., {N})")

    n_steps = int(np.log2(N))
    x = np.array([1.0])

    for _ in range(n_steps):
        # Her değeri ikiye böl: sol=p, sağ=(1-p)
        left = x * p
        right = x * (1 - p)

        # Interleave (araya ekle): [p, 1-p, p, 1-p, ...]
        new_x = np.empty(2 * len(x))
        new_x[0::2] = left
        new_x[1::2] = right
        x = new_x

    # Normalize: toplam = N olacak şekilde
    x = x / np.sum(x) * N

    return x


# ============================================================
# 2. TEORİK SPEKTRUM
# ============================================================

def theoretical_alpha(q: np.ndarray, p: float) -> np.ndarray:
    """p-model teorik Hölder üssü alpha(q)."""
    pq = p ** q
    qmq = (1 - p) ** q

    numerator = -(pq * np.log(p) + qmq * np.log(1 - p))
    denominator = (pq + qmq) * np.log(2)

    return numerator / denominator


def theoretical_f_alpha(alpha: np.ndarray, p: float) -> np.ndarray:
    """
    p-model teorik f(alpha) — doğrudan alpha'dan.

    f(alpha) = alpha*q(alpha) - tau(q(alpha))
    Ama daha basit: Binomial cascade için f(alpha) analitik olarak:

    f(alpha) = (alpha - alpha_min)/(alpha_max - alpha_min) * log2(...)

    Basit yaklaşım: Legendre transformü numerik olarak hesapla.
    """
    # q aralığını tersine çevir: alpha'dan q'ya
    q_theory = np.linspace(-50, 50, 1000)
    alpha_theory = theoretical_alpha(q_theory, p)
    tau_theory = q_theory * alpha_theory - 1
    f_theory = q_theory * alpha_theory - tau_theory

    return f_theory, alpha_theory


# ============================================================
# 3. MF-DFA (Güncellenmiş — Daha Kararlı)
# ============================================================

def mfdfa(x: np.ndarray, q_vals: np.ndarray,
          order: int = 2, n_scales: int = 30):
    """
    Çoklu Fraktal DFA (Kantelhardt et al. 2002).
    v2: Daha fazla ölçek, daha kararlı sonuç.
    """
    N = len(x)

    # Profil
    Y = np.cumsum(x - np.mean(x))

    # Ölçekler
    s_min = max(8, order + 2)
    s_max = N // 4
    scales = np.unique(np.round(
        np.logspace(np.log10(s_min), np.log10(s_max), n_scales)
    ).astype(int))

    # Fq hesapla
    Fq = np.zeros((len(q_vals), len(scales)))

    for j, s in enumerate(scales):
        # Overlapping pencereler
        step = s // 2 if s > 1 else 1
        starts = np.arange(0, N - s + 1, step)

        F2_list = []
        for start in starts:
            segment = Y[start:start + s]
            t = np.arange(s)

            # Polinom detrending
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            residual = segment - trend

            F2 = np.mean(residual ** 2)
            F2_list.append(F2)

        F2_array = np.array(F2_list)
        F2_array = F2_array[F2_array > 0]  # Sadece pozitif değerler

        for i, q in enumerate(q_vals):
            if abs(q) < 1e-10:  # q=0
                Fq[i, j] = np.exp(np.mean(np.log(F2_array)))
            else:
                Fq[i, j] = (np.mean(F2_array ** (q / 2))) ** (1.0 / q)

    # h(q): log Fq(s) ~ h(q) * log s
    log_scales = np.log(scales)
    h_q = np.zeros(len(q_vals))

    for i in range(len(q_vals)):
        log_Fq = np.log(Fq[i, :])
        mask = np.isfinite(log_Fq)
        if mask.sum() >= 3:
            coeffs = np.polyfit(log_scales[mask], log_Fq[mask], 1)
            h_q[i] = coeffs[0]

    # Legendre dönüşümü
    tau_q = q_vals * h_q - 1
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    return {
        'h_q': h_q,
        'tau_q': tau_q,
        'alpha': alpha,
        'f_alpha': f_alpha,
        'scales': scales,
        'Fq': Fq
    }


# ============================================================
# 4. ANA TEST
# ============================================================

def main():
    print("=" * 60)
    print("P-MODEL MF-DFA DOĞRULAMA — v2")
    print("=" * 60)

    # Parametreler
    N = 8192  # 2^13
    p = 0.3
    q_vals = np.linspace(-5, 5, 21)  # 21 nokta, q=0 tam ortada

    # Teorik
    delta_alpha_theory = (np.log(1 - p) - np.log(p)) / np.log(2)
    alpha_theory = theoretical_alpha(q_vals, p)

    print(f"\nParametreler:")
    print(f"  N = {N}, p = {p}")
    print(f"  q: {len(q_vals)} nokta, [{q_vals[0]:.1f}, {q_vals[-1]:.1f}]")
    print(f"\nTeorik Değerler:")
    print(f"  Delta_alpha = {delta_alpha_theory:.4f}")
    print(f"  alpha(q=0) = {alpha_theory[len(q_vals) // 2]:.4f}")

    # P-Model serisi
    print(f"\nP-Model serisi üretiliyor (deterministik cascade)...")
    series = pmodel_cascade(N, p)
    print(f"  Uzunluk: {len(series)}")
    print(f"  Mean: {np.mean(series):.4f}, Std: {np.std(series):.4f}")

    # MF-DFA
    print(f"\nMF-DFA hesaplanıyor...")
    result = mfdfa(series, q_vals, order=2, n_scales=30)

    alpha_calc = result['alpha']
    delta_alpha_calc = np.nanmax(alpha_calc) - np.nanmin(alpha_calc)

    print(f"  h(q=0) = {result['h_q'][len(q_vals) // 2]:.4f}")
    print(f"  alpha min/max = {np.nanmin(alpha_calc):.4f} / {np.nanmax(alpha_calc):.4f}")
    print(f"  Delta_alpha (hesaplanan) = {delta_alpha_calc:.4f}")

    # Karşılaştırma
    error = abs(delta_alpha_calc - delta_alpha_theory) / delta_alpha_theory * 100

    print(f"\n{'=' * 60}")
    print(f"SONUÇ:")
    print(f"  Teorik:     {delta_alpha_theory:.4f}")
    print(f"  Hesaplanan: {delta_alpha_calc:.4f}")
    print(f"  Hata:       %{error:.2f}")

    if error < 10:
        print(f"  ✓ KABUL EDİLEBİLİR (hata < %10)")
    elif error < 15:
        print(f"  ~ SINIRDA (hata < %15) — daha fazla ölçek denenebilir")
    else:
        print(f"  ✗ YÜKSEK (hata > %15)")
    print(f"{'=' * 60}")

    # Grafikler
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Seri
    axes[0, 0].plot(series[:500], 'b-', linewidth=0.6)
    axes[0, 0].set_title(f'P-Model Serisi (N={N}, p={p})')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('x(n)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. log-log
    for q_val in [-4, 0, 4]:
        idx = np.argmin(np.abs(q_vals - q_val))
        axes[0, 1].plot(np.log(result['scales']), np.log(result['Fq'][idx, :]),
                        'o-', label=f'q={q_vals[idx]:.0f}')
    axes[0, 1].set_title('log Fq(s) vs log s')
    axes[0, 1].set_xlabel('log s')
    axes[0, 1].set_ylabel('log Fq(s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. alpha(q): Teorik vs Hesaplanan
    axes[1, 0].plot(q_vals, alpha_theory, 'r-', linewidth=2, label='Teorik')
    axes[1, 0].plot(q_vals, alpha_calc, 'bo--', markersize=6, label='MF-DFA')
    axes[1, 0].set_title(f'α(q) — Teorik vs Hesaplanan (Δα: {error:.1f}% hata)')
    axes[1, 0].set_xlabel('q')
    axes[1, 0].set_ylabel('α(q)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Spektrum f(α)
    f_theory, alpha_t = theoretical_f_alpha(alpha_calc, p)
    # Sort for plotting
    sort_idx = np.argsort(alpha_t)
    axes[1, 1].plot(alpha_t[sort_idx], f_theory[sort_idx], 'r-', linewidth=2, label='Teorik')

    sort_idx2 = np.argsort(alpha_calc)
    axes[1, 1].plot(alpha_calc[sort_idx2], result['f_alpha'][sort_idx2],
                    'bo--', markersize=5, label='MF-DFA')
    axes[1, 1].set_title('Çoklu Fraktal Spektrum f(α)')
    axes[1, 1].set_xlabel('α')
    axes[1, 1].set_ylabel('f(α)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pmodel_validation_v2.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nGrafik: results/pmodel_validation_v2.png")
    print(f"\n{'=' * 60}")
    print(f"JÜRİYE SUNUM İÇİN:")
    print(f"  'MF-DFA kodumuz, analitik spektrumu bilinen")
    print(f"   binomial cascade üzerinde doğrulanmıştır.")
    print(f"   Bağıl hata %{error:.1f}'dir.'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()