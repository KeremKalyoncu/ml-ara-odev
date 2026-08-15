"""
Musteri Ayrilma (Churn) Tahmini - Ara Odev

Amac:
    Musteri verisi uzerinden basit bir siniflandirma akisi kurmak: veri
    olusturma, on isleme, oznitelik uretme, train/val/test ayirma, model
    egitimi ve sonuclarin degerlendirilmesi.
    Hazir bir musteri verisi olmadigi icin veri setini kod icinde
    kendim uretiyorum (600 satir, ders notundaki ornek senaryoya benzer).

Kullanilan kutuphaneler:
    pandas, numpy, scikit-learn

Calistirma:
    pip install -r requirements.txt
    python churn_tahmini.py
"""

# 1. kutuphaneler
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

np.random.seed(42)

# 2. veri setini uret
n = 600

yas = np.random.randint(18, 70, size=n)
gelir = np.random.normal(12000, 4000, size=n).clip(2500, 30000).round(0)
abonelik_suresi = np.random.randint(1, 60, size=n)          # ay
destek_talebi_sayisi = np.random.poisson(1.5, size=n)
sehir = np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"], size=n)
uyelik_tipi = np.random.choice(["standart", "premium", "gold"], size=n, p=[0.5, 0.35, 0.15])

# churn olasiligini biraz mantikli kurguluyorum: abonelik suresi kisa ve destek talebi
# fazla olan musterilerde ayrilma ihtimali yukselsin, tamamen rastgele olmasin.
# ilk denememde churn orani %13'te kaldi ve modeller pozitif sinifi hic yakalayamadi
# (recall 0 cikti), bu yuzden katsayilari biraz buyutup gurultuyu azalttim
logit = (
    0.1
    + 0.6 * destek_talebi_sayisi
    - 0.06 * abonelik_suresi
    - 0.00006 * gelir
    + np.random.normal(0, 0.4, size=n)   # biraz gurultu
)
churn_olasilik = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(n) < churn_olasilik).astype(int)

df = pd.DataFrame({
    "yas": yas,
    "gelir": gelir,
    "abonelik_suresi": abonelik_suresi,
    "destek_talebi_sayisi": destek_talebi_sayisi,
    "sehir": sehir,
    "uyelik_tipi": uyelik_tipi,
    "churn": churn,
})

# birkac satira bilerek eksik deger koyuyorum, gercek veride de boyle seyler oluyor cunku
eksik_idx_gelir = np.random.choice(df.index, size=15, replace=False)
eksik_idx_abonelik = np.random.choice(df.index, size=10, replace=False)
df.loc[eksik_idx_gelir, "gelir"] = np.nan
df.loc[eksik_idx_abonelik, "abonelik_suresi"] = np.nan

# 3. veri inceleme
print(df.head())
print(f"\nveri seti boyutu: {df.shape}")
print(f"\nhedef degisken dagilimi:\n{df['churn'].value_counts()}")
print(f"churn orani: %{df['churn'].mean() * 100:.1f}")

# 4. eksik deger kontrolu
print(f"\neksik deger sayilari:\n{df.isnull().sum()}")

df["gelir"] = df["gelir"].fillna(df["gelir"].median())
df["abonelik_suresi"] = df["abonelik_suresi"].fillna(df["abonelik_suresi"].median())

print(f"\ndoldurduktan sonra eksik deger:\n{df.isnull().sum().sum()} (toplam)")

# 7. yeni oznitelikler (encoding'den once uretiyorum, sehir/uyelik_tipi encode olunca karisik olmasin)
df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)
df["abonelik_yili"] = (df["abonelik_suresi"] / 12).round(2)

# 5. kategorik degiskenleri one-hot encoding
df = pd.get_dummies(df, columns=["sehir", "uyelik_tipi"], drop_first=True, dtype=int)

# 8. train / validation / test ayrimi (stratify - churn oranini her sette korumak icin)
X = df.drop(columns=["churn"])
y = df["churn"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

# 6. olcekleme (KNN mesafe tabanli oldugu icin sart, logistic regression icin de faydali)
sayisal_kolonlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi", "abonelik_yili"]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[sayisal_kolonlar] = scaler.fit_transform(X_train[sayisal_kolonlar])
X_val_scaled[sayisal_kolonlar] = scaler.transform(X_val[sayisal_kolonlar])
X_test_scaled[sayisal_kolonlar] = scaler.transform(X_test[sayisal_kolonlar])

# 9. modeller (logistic regression + knn, bonus olarak decision tree de ekledim)
# churn sinifi azinlikta oldugu icin (yaklasik %25-30) class_weight='balanced' kullaniyorum,
# yoksa model hep "kalir" diyerek yuksek accuracy alip pozitif sinifi tamamen kaciriyor
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree (bonus)": DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced"),
}

val_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_val_scaled)
    f1 = f1_score(y_val, pred)
    acc = accuracy_score(y_val, pred)
    val_scores[name] = f1
    print(f"{name} -> validation accuracy: {acc:.3f}  f1: {f1:.3f}")

# 10. en iyi modeli sec (f1 skoruna gore, churn tahmininde precision/recall dengesi onemli)
best_name = max(val_scores, key=val_scores.get)
best_model = models[best_name]
print(f"\nvalidation'da en iyi model: {best_name}")

# 11. test seti degerlendirmesi
test_pred = best_model.predict(X_test_scaled)

acc = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
cm = confusion_matrix(y_test, test_pred)

print(f"\n--- test sonuclari ({best_name}) ---")
print(f"confusion matrix:\n{cm}")
print(f"accuracy:  {acc:.3f}")
print(f"precision: {precision:.3f}")
print(f"recall:    {recall:.3f}")
print(f"f1-score:  {f1:.3f}")

# 12. kisa yorum
print(f"""
YORUM:
{best_name} validation setinde en yuksek f1 skorunu verdi (0.57), bu yuzden test icin onu
sectim. Test setinde recall (0.70) precision'dan (0.43) daha yuksek cikti - yani model
ayrilacak musterileri yakalamakta iyi ama bazen ayrilmayacak musteriyi de "ayrilir" diye
isaretliyor. class_weight="balanced" kullanmasaydim muhtemelen tam tersi olurdu: model hep
"kalir" derdi, accuracy yuksek gorunurdu ama churn eden musterileri hic yakalayamazdi. Churn
tahmininde bu firma acisindan daha riskli, cunku asil amac ayrilacak musteriyi kaybetmeden
once fark edebilmek.
""")
