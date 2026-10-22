"""MF-DFA duzeltmesini dogrula: h(q=2) ve Delta_alpha, bilinen sureclerde."""
import sys
import numpy as np
sys.path.insert(0, r"C:\TUBITAK_Proje")
from mfdfa_core import (mfdfa, spectrum_width, pmodel_cascade, theoretical_alpha,
                        fgn, fbm, white_noise)

Q = np.linspace(-5, 5, 21)
i2 = int(np.where(np.isclose(Q, 2.0))[0][0])
rng = np.random.default_rng(0)


print("=== DUZELTME DOGRULAMASI ===")
r = mfdfa(rng.standard_normal(8192), Q)
print(f"beyaz gurultu : h(q=2)={r['h_q'][i2]:.4f} (teorik 0.5)  DA={spectrum_width(r):.4f}")

for H in (0.3, 0.5, 0.7, 0.9):
    r = mfdfa(fgn(8192, H, rng), Q)
    print(f"fGn H={H}     : h(q=2)={r['h_q'][i2]:.4f} (teorik {H})  DA={spectrum_width(r):.4f}")

print()
for p in (0.4, 0.3, 0.2):
    x = pmodel_cascade(8192, p)
    r = mfdfa(x, Q)
    theo = theoretical_alpha(Q, p)
    print(f"p-model p={p} : h(q=2)={r['h_q'][i2]:.4f}  "
          f"DA={spectrum_width(r):.4f} (teorik DA={theo.max()-theo.min():.4f})")
