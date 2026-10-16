"""
Deprem verisi faz analizi: akut vs normalleşme dönemlerini belirleme.
"""
import pandas as pd
import numpy as np
import os

base = r'C:\TUBITAK_Proje\data\earthquake\turkish'
files = ['depremGun1tr.csv', 'depremGun2tr.csv', 'depremGun3tr.csv',
         'depremGun4tr.csv', 'depremGun5tr.csv']

frames = []
for f in files:
    df = pd.read_csv(os.path.join(base, f), usecols=['Date'])
    frames.append(df)

all_d = pd.concat(frames, ignore_index=True)
all_d['Date'] = pd.to_datetime(all_d['Date'], utc=True)
all_d = all_d.set_index('Date').sort_index()

hourly = all_d.resample('1h').size()

print('=' * 70)
print('SAATLIK TWEET HACMI')
print('=' * 70)
for ts, v in hourly.items():
    bar = '#' * int(v / 2000)
    print(f'{ts.strftime("%d %b %H:%M")} | {v:>7,} {bar}')

print()
print('=' * 70)
print('GUNLUK OZET')
print('=' * 70)
daily = all_d.resample('1D').size()
for ts, v in daily.items():
    print(f'{ts.strftime("%d %b")} | {v:>9,} tweet')

print()
print(f'TOPLAM: {len(all_d):,} tweet')
print(f'TOPLAM SURE: {(all_d.index.max() - all_d.index.min())}')
