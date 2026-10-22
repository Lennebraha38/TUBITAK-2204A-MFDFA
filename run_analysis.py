"""
Tam analiz surucusu: her olay icin seri uret + secilen sinyallerde MF-DFA.

Kullanim: python run_analysis.py [n_surr]
"""
import subprocess
import sys
import os

PY = sys.executable
ROOT = r"C:\TUBITAK_Proje"
n_surr = sys.argv[1] if len(sys.argv) > 1 else "100"

EVENTS = [
    # Kahramanmaras depremi: 2023-02-06 04:17 TRT = 01:17 UTC
    dict(prefix="km2023",
         glob="results\\scored2\\deprem*_score2.csv.gz",
         datecol="Date", acute="2023-02-06T01:17"),
    # Izmir depremi: 2020-10-30 14:51 TRT = 11:51 UTC
    dict(prefix="izmir2020",
         glob="results\\scored2\\izmir_earthquake_score2.csv.gz",
         datecol="Datetime", acute="2020-10-30T11:51"),
]

# Ana sinyal + robustness sinyalleri
SIGNALS = ["distress", "sadness", "fear", "anger", "joy", "valence_emo", "val_sav"]


def run(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"!!! HATA (kod {r.returncode}): {cmd}", flush=True)


for ev in EVENTS:
    prefix = ev["prefix"]
    scsv = os.path.join(ROOT, "results", f"{prefix}_series_1min.csv")
    if not os.path.exists(scsv):
        run([PY, "-u", "build_series2.py", ev["glob"], ev["datecol"], prefix])
    else:
        print(f"Seri var, atlandi: {scsv}", flush=True)

    for sig in SIGNALS:
        outcsv = os.path.join(ROOT, "results", f"{prefix}_{sig}_mfdfa.csv")
        if os.path.exists(outcsv):
            print(f"Analiz var, atlandi: {outcsv}", flush=True)
            continue
        run([PY, "-u", "analyze2.py", scsv, prefix, sig, ev["acute"], "24", n_surr])

print("\n=== TUM ANALIZ BITTI ===", flush=True)
