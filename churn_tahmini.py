"""
Musteri Ayrilma (Churn) Tahmini - Ara Odev

Amac:
    Musteri verisi uzerinden basit bir siniflandirma akisi kurmak.
    Veri seti hazir olmadigi icin kendim uretiyorum (en az 100 satir).

Kullanilan kutuphaneler:
    pandas, numpy, scikit-learn

Nasil calisir:
    python churn_tahmini.py
"""

# 1. kutuphaneler
import numpy as np
import pandas as pd

# TODO: veri setini uret / oku (yas, gelir, abonelik_suresi, destek_talebi_sayisi, sehir, uyelik_tipi, churn)
# TODO: veri inceleme (head, shape, hedef degisken dagilimi)
# TODO: eksik deger kontrolu
# TODO: kategorik degiskenleri encode et
# TODO: olcekleme
# TODO: en az 1 yeni oznitelik uret
# TODO: train / validation / test ayir (stratify)
# TODO: Logistic Regression + KNN egit, karsilastir
# TODO: en iyi modeli test setinde degerlendir (confusion matrix, accuracy, precision, recall, f1)
# TODO: kisa yorum
