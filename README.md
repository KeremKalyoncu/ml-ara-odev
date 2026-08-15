# Musteri Ayrilma (Churn) Tahmini - ML Ara Odev

Turkiye Yapay Zeka Akademisi Makine Ogrenmesi dersi ara odevi.

## Amac
Musteri verisi uzerinde temel bir siniflandirma akisi kurmak: veri olusturma,
on isleme, oznitelik uretme, train/validation/test ayirma, model egitimi ve
sonuclarin degerlendirilmesi.

## Veri seti
Hazir bir musteri verisi olmadigi icin veri seti kod icinde uretiliyor
(en az 100 satir). Sutunlar: yas, gelir, abonelik_suresi, destek_talebi_sayisi,
sehir, uyelik_tipi, churn (0 = kalir, 1 = ayrilir).

## Nasil calistirilir
```bash
pip install -r requirements.txt
python churn_tahmini.py
```

## Sonuc

Validation karsilastirmasi (f1-score):

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 0.750 | 0.571 |
| KNN | 0.775 | 0.400 |
| Decision Tree (bonus) | 0.708 | 0.507 |

Logistic Regression secildi (`class_weight="balanced"` ile), test sonuclari:

- Accuracy: 0.725
- Precision: 0.432
- Recall: 0.704
- F1-score: 0.535

Churn azinlik sinif oldugu icin (%22.7) class_weight kullanmadan model hep "kalir" derdi.
Recall'un precision'dan yuksek cikmasi churn tahmininde tercih edilir sekilde bir sonuc -
amac ayrilacak musteriyi kacirmamak.
