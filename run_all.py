"""
==============================================================================
TUBITAK 2204-A  -  TEK GIRIS NOKTASI (Uctan uca boru hatti)
==============================================================================
Asama sirasi:

  0) VERI      : Kaggle'dan indirilen ham CSV'ler  -> data/
  1) SKORLAMA  : duygu modeli (furkankarakuz) + savasy  -> results/scored2/
  2) SERI      : 1 dakikalik duygu serileri            -> results/*_series_1min.csv
  3) MF-DFA    : sinyal x pencere MF-DFA + surrogalar -> results/*_mfdfa.csv|png
  4) KALIBR.   : bilinen surecler icin Delta_alpha cetveli
  5) KARSIL.   : iki olayin karsilastirma tablosu/figuru
  6) TEST      : pytest ile matematik dogrulama

Kullanim:
    python run_all.py                 # eksik olan tum asamalari calistirir
    python run_all.py --stage 3       # sadece 3. asama
    python run_all.py --force         # var olan ciktilari da yeniden hesapla
    python run_all.py --n-surrogate 200

NOT: 1) ve 3) asamalari GPU + indirilmis veri gerektirir; diger asamalar
     saf CPU'dur. Var olan cikti dosyalari ATLANIR (yeniden baslatilabilirlik).
==============================================================================
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Olay tanimlari ---------------------------------------------------------
EVENTS = [
    # Kahramanmaras depremi: 2023-02-06 04:17 TRT = 01:17 UTC
    dict(prefix="km2023",
         glob=os.path.join("results", "scored2", "deprem*_score2.csv.gz"),
         datecol="Date",
         acute="2023-02-06T01:17",
         score_files=["depremGun{i}tr.csv".format(i=i) for i in range(1, 6)]),
    # Izmir depremi: 2020-10-30 14:51 TRT = 11:51 UTC
    dict(prefix="izmir2020",
         glob=os.path.join("results", "scored2", "izmir_earthquake_score2.csv.gz"),
         datecol="Datetime",
         acute="2020-10-30T11:51",
         score_files=["izmir_earthquake.csv"]),
]

# Ana sinyal + sağlamlık (robustness) sinyalleri
SIGNALS = ["distress", "sadness", "fear", "anger", "joy", "valence_emo", "val_sav"]

STAGE_NAMES = ["veri", "skorlama", "seri", "mfdfa", "kalibrasyon", "karsilastirma", "test"]


# ---------------------------------------------------------------------------
def sh(cmd, desc, must=False):
    """Alt komutu calistirir; hata olursa (must=True ise) durur."""
    print("\n" + "=" * 78, flush=True)
    print(">>> " + desc, flush=True)
    print("    " + " ".join(cmd), flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"!!! HATA ({r.returncode}) [{dt:.0f}s]: {desc}", flush=True)
        if must:
            raise SystemExit(r.returncode)
        return False
    print(f"--- TAMAM [{dt:.0f}s]: {desc}", flush=True)
    return True


def p(*parts):
    return os.path.join(ROOT, *parts)


# ---------------------------------------------------------------------------
def stage_veri(args):
    """0) Ham verinin varligini denetler (indirme kullanici tarafindan yapilir)."""
    ok = True
    for ev in EVENTS:
        for f in ev["score_files"]:
            src = p("data", "earthquake", "turkish", f) if ev["prefix"] == "km2023" \
                else p("data", "izmir", f)
            if os.path.exists(src):
                print(f"  [VAR]  {src}", flush=True)
            else:
                print(f"  [YOK]  {src}  <- Kaggle'dan indirilip data/ altina konmali",
                      flush=True)
                ok = False
    if not ok:
        print("\n  !! Ham veri eksik. Surrogate/analiz asamalari yine de denenebilir.",
              flush=True)
    return ok


def stage_skorlama(args):
    """1) Duygu modeli ile tweet'leri skorlama (GPU gerekir)."""
    for ev in EVENTS:
        sh([PY, "-u", "score2.py", ev["prefix"]],
           f"1) Skorlama: {ev['prefix']}", must=False)


def stage_seri(args):
    """2) 1 dakikalik duygu serilerinin uretilmesi."""
    for ev in EVENTS:
        out = p("results", f"{ev['prefix']}_series_1min.csv")
        if os.path.exists(out) and not args.force:
            print(f"  [ATLANDI] {out}", flush=True)
            continue
        sh([PY, "-u", "build_series2.py", ev["glob"], ev["datecol"], ev["prefix"]],
           f"2) Seri: {ev['prefix']}", must=False)


def stage_mfdfa(args):
    """3) MF-DFA analizi (sinyal x pencere) + surrogalar."""
    for ev in EVENTS:
        scsv = p("results", f"{ev['prefix']}_series_1min.csv")
        if not os.path.exists(scsv):
            print(f"  !! {scsv} yok, {ev['prefix']} atlandi", flush=True)
            continue
        for sig in SIGNALS:
            outcsv = p("results", f"{ev['prefix']}_{sig}_mfdfa.csv")
            if os.path.exists(outcsv) and not args.force:
                print(f"  [ATLANDI] {outcsv}", flush=True)
                continue
            sh([PY, "-u", "analyze2.py", scsv, ev["prefix"], sig,
                ev["acute"], "24", str(args.n_surrogate)],
               f"3) MF-DFA: {ev['prefix']} / {sig}", must=False)


def stage_kalibrasyon(args):
    """4) Delta_alpha kalibrasyon cetveli."""
    sh([PY, "-u", "calibration.py"], "4) Kalibrasyon cetveli", must=False)


def stage_karsilastirma(args):
    """5) Iki olayin karsilastirmasi."""
    sh([PY, "-u", "compare_events2.py"], "5) Karsilastirma tablosu/figuru", must=False)


def stage_test(args):
    """6) Matematik dogrulama testleri."""
    sh([PY, "-m", "pytest", "tests/", "-q", "--no-header"], "6) pytest", must=False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TUBITAK 2204-A uctan uca boru hatti")
    ap.add_argument("--stage", type=int, default=None,
                    help="sadece tek asamayi calistir (0-6)")
    ap.add_argument("--force", action="store_true",
                    help="var olan ciktilari da yeniden hesapla")
    ap.add_argument("--n-surrogate", type=int, default=100,
                    help="her sinyal icin surrogal sayisi (varsayilan 100)")
    args = ap.parse_args()

    print("=" * 78)
    print("TUBITAK 2204-A  |  MF-DFA  |  Uctan uca boru hatti")
    print("=" * 78)
    print(f"  Python  : {sys.version.split()[0]}  ({PY})")
    print(f"  KOK     : {ROOT}")
    print(f"  Mod     : {'--force' if args.force else 'yalnizca eksikler'}"
          f" | surrogal={args.n_surrogate}")

    stages = [
        ("0) Veri denetimi", stage_veri),
        ("1) Skorlama", stage_skorlama),
        ("2) 1-dk seri uretimi", stage_seri),
        ("3) MF-DFA analizi", stage_mfdfa),
        ("4) Kalibrasyon cetveli", stage_kalibrasyon),
        ("5) Karsilastirma", stage_karsilastirma),
        ("6) Testler", stage_test),
    ]

    t0 = time.time()
    for i, (name, fn) in enumerate(stages):
        if args.stage is not None and i != args.stage:
            continue
        print(f"\n########## ASAMA {i} : {name} ##########", flush=True)
        try:
            fn(args)
        except Exception as e:            # noqa: BLE001
            print(f"!!! ASAMA {i} HATA: {e}", flush=True)
            if args.stage is not None:
                raise

    print("\n" + "=" * 78)
    print(f"BORU HATTI TAMAMLANDI  ({(time.time()-t0)/60:.1f} dk)")
    print("=" * 78)
    print("Ciktilar:")
    print("  results\\*_series_1min.csv        1-dk duygu serileri")
    print("  results\\*_mfdfa.csv / .png       MF-DFA sonuclari ve figurler")
    print("  results\\calibration_table.csv    Delta_alpha kalibrasyon cetveli")
    print("  results\\comparison_all.csv       iki olayin karsilastirma tablosu")
    print("  results\\comparison_summary.txt   duz metin ozet")
    print("  results\\comparison_plot.png      karsilastirma figuru")


if __name__ == "__main__":
    main()
