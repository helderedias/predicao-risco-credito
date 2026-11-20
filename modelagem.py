import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

# 1. Simulação da base de dados com correlação real de risco
np.random.seed(42)
data_size = 10000

idade = np.random.randint(18, 70, size=data_size)
renda_anual = np.random.normal(55000, 15000, size=data_size)
historico_ruim = np.random.choice([0, 1], size=data_size, p=[0.85, 0.15])

score = (renda_anual / 55000) * 2 - (historico_ruim * 3) + np.random.normal(0, 1, size=data_size)
limiar = np.percentile(score, 18)
risco_credito = np.where(score <= limiar, 1, 0)

df = pd.DataFrame({
    'idade': idade,
    'renda_anual': renda_anual,
    'tempo_emprego': np.random.choice([np.nan, 2, 5, 10, 15], size=data_size, p=[0.05, 0.30, 0.35, 0.20, 0.10]),
    'valor_emprestimo': np.random.normal(15000, 5000, size=data_size),
    'tipo_moradia': np.random.choice(['PROPRIA', 'ALUGADA', 'HIPOTECA'], size=data_size),
    'risco_credito': risco_credito
})

# 2. Pré-processamento e Pipeline de Dados
df['tempo_emprego'] = df['tempo_emprego'].fillna(df['tempo_emprego'].median())
df = pd.get_dummies(df, columns=['tipo_moradia'], drop_first=True)

X = df.drop(columns=['risco_credito'])
y = df['risco_credito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Treinamento dos Modelos
model_lr = LogisticRegression(random_state=42)
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=7)

model_lr.fit(X_train_scaled, y_train)
model_dt.fit(X_train, y_train)
model_knn.fit(X_train_scaled, y_train)

# 4. Avaliação no Terminal
modelos = [
    ("Regressão Logística", model_lr, X_test_scaled),
    ("Árvore de Decisão", model_dt, X_test),
    ("k-NN", model_knn, X_test_scaled)
]

for nome, modelo, dados_teste in modelos:
    preds = modelo.predict(dados_teste)
    probs = modelo.predict_proba(dados_teste)[:, 1]
    print(f"\n--- {nome} ---")
    print(classification_report(y_test, preds))
    print(f"AUC-ROC: {roc_auc_score(y_test, probs):.3f}")