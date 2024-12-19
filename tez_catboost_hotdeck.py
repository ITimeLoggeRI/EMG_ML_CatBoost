import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier  # CatBoost sınıflandırıcıyı ekledim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Veri dosyasının yolu
file_path = r'C:\Users\pc\Desktop\Coding edu\Tez ai\other_version\ktsverivesonuc.csv'

# CSV dosyasını pandas ile oku
df = pd.read_csv(file_path, encoding='ISO-8859-9', delimiter=';')

# Hedef değişken ve bağımsız değişkenleri ayır
X = df.drop('Ana el', axis=1)  # Bağımsız değişkenler
y = df['Ana el']  # Hedef değişken

# Virgülleri nokta ile değiştir ve sayısal değerlere dönüştür
X = X.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
X = X.astype(float)

# Hedef değişkeni sayısal değerlere dönüştür (eğer kategorik ise)
y = y.astype(str)

def hot_deck_impute(df):
    for column in df.columns:
        missing = df[column].isnull()
        non_missing = df.loc[~missing, column]
        if missing.any():
            if len(non_missing) > 0:
                df.loc[missing, column] = non_missing.sample(missing.sum(), replace=True).values
            else:
                df.loc[missing, column] = df[column].mean()
    return df

# X için Hot Deck Imputation uygulama
X_imputed = hot_deck_impute(X.copy())

# NaN kontrolü
if X_imputed.isnull().values.any():
    print("X_imputed contains NaN values.")

# Hedef değişken için Label Encoding yapalım
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Hedef değişkende eksik değerler varsa bunları da Hot Deck Imputation ile dolduralım
y_imputed_df = pd.DataFrame(y_encoded, columns=['Ana el'])
y_imputed = hot_deck_impute(y_imputed_df).values.ravel()

# NaN kontrolü
if np.isnan(y_imputed).any():
    print("y_imputed contains NaN values.")

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# CatBoost modelini oluştur
model = CatBoostClassifier(random_state=42, verbose=0)

# Modeli eğit
model.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
