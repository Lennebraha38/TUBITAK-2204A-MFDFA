"""
MF-DFA kalibrasyon: bilinen surecler icin Delta_alpha "cetveli".
Juriye "1.76 buyuk mu?" sorusunun yaniti.
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import mfdfa, spectrum_width, pmodel_cascade, surrogate_test

N = 8192
Q = np.linspace(-5, 5, 21)
rng = np.random.default_rng(0)


def fbm(N, H, rng):
    """Spektral sentez ile fraksiyonel Brownian motion."""
    n = N
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-12
    S = freqs ** (-(2 * H + 1))
    S[0] = 0
    amp = np.sqrt(S)
    ph = rng.uniform(0, 2 * np.pi, len(amp))
    f = np.fft.irfft(amp * np.exp(1j * ph), n=n)
    return np.cumsum(f)


def report(name, x, do_surr=False):
    r = mfdfa(x, Q, order=2, n_scales=30)
    da = spectrum_width(r)
    line = f"{name:22s} N={len(x):5d}  DA={da:7.4f}  h(q=2)={r['h_q'][10]:6.3f}"
    if do_surr:
        st = surrogate_test(x, Q, n_surr=50, kind="iaaft", seed=1)
        line += f"  IAAFT z={st['z']:+.2f}"
    print(line, flush=True)


print("=" * 78)
print("MF-DFA KALIBRASYON CETVELI (N=8192, q in [-5,5], 30 olcek)")
print("=" * 78)
print(f"{'Surec':22s} {'N':>5s}  {'Delta_alpha':>8s}  {'h(q=2)':>6s}")
print("-" * 78)

report("Beyaz gurultu", rng.standard_normal(N))
report("Tekduze dagilim", rng.uniform(0, 1, N))
report("Rastgele yuruyus", np.cumsum(rng.standard_normal(N)))
report("fBm H=0.3", fbm(N, 0.3, rng))
report("fBm H=0.5", fbm(N, 0.5, rng))
report("fBm H=0.7", fbm(N, 0.7, rng))
report("fBm H=0.9", fbm(N, 0.9, rng))

# AR(1) surecleri
for phi in (0.5, 0.9, 0.98):
    e = rng.standard_normal(N)
    ar = np.zeros(N)
    for i in range(1, N):
        ar[i] = phi * ar[i - 1] + e[i]
    report(f"AR(1) phi={phi}", ar)

# p-model (multifraktal referans)
for p in (0.5, 0.4, 0.3, 0.2):
    report(f"p-model p={p}", pmodel_cascade(N, p))

print("-" * 78)
print("Not: Delta_alpha ~ 0.05-0.15 monofraktal (beyaz gurultu, fBm);")
print("     p-model gibi multifraktal sureclerde 1.0-1.3;")
print("     gozlemlenen duygu serisi degerleri bu cetvelle karsilastirilir.")
