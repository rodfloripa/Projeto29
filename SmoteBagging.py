
import os
os.system("pip install ucimlrepo imblearn xgboost tqdm scikit-learn --quiet")

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier # Mantido para referência, mas não será usado diretamente
from sklearn.ensemble import RandomForestClassifier # Usaremos RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.impute import SimpleImputer
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed # Usar ProcessPoolExecutor para tqdm
from ucimlrepo import fetch_ucirepo
from tqdm.notebook import tqdm # Para a barra de progresso
import tensorflow as tf # Mantido caso precise de referência, mas não será usado para detecção de GPU aqui
import time

# 1. Carregar dados
ozone = fetch_ucirepo(id=172)
X = ozone.data.features.to_numpy()
y = ozone.data.targets.to_numpy().ravel()
print("Dados carregados. Shape X: {}, Shape Y: {}".format(X.shape, y.shape))

# 2. Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Dados divididos. Shape X_train: {}, Shape X_test: {}".format(X_train.shape, X_test.shape))

# 3. Tratar NaNs com imputação
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test) # só transform
print("NaNs tratados. Iniciando o treinamento do ensemble...")

# O RandomForestClassifier do scikit-learn roda na CPU por padrão.
print("O treinamento do RandomForestClassifier será executado na CPU, pois esta implementação não utiliza GPU.")

# Função para treinar o modelo
def treinar_modelo(i, X_train_arg, y_train_arg):
    smote = SMOTE(random_state=i, sampling_strategy=0.5)
    X_res, y_res = smote.fit_resample(X_train_arg, y_train_arg)
    # Usar RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=i, n_estimators=100) # n_estimators é um bom ponto de partida
    rf_model.fit(X_res, y_res)
    # Remover time.sleep, pois o treinamento agora é real e o RF é rápido
    return rf_model

def criar_ensemble(n_modelos):
    n_processos = min(os.cpu_count(), n_modelos)
    modelos = []
    with ProcessPoolExecutor(max_workers=n_processos) as executor:
        futures = {executor.submit(treinar_modelo, i, X_train, y_train): i for i in range(n_modelos)}

        for future in tqdm(as_completed(futures), total=n_modelos, desc="Treinando Modelos"): # Usar as_completed com tqdm
            modelos.append(future.result())
    return modelos

# Criar os modelos
modelos = criar_ensemble(6)

print("4. Realizando previsões com o ensemble...")
def prever_ensemble(X_data):
    todas_preds = np.array([model.predict(X_data) for model in modelos])
    # Para Random Forest, o threshold padrão é 0.5. Poderíamos usar predict_proba para ajustar o threshold se necessário.
    return (np.mean(todas_preds, axis=0) > 0.5).astype(int)

y_pred = prever_ensemble(X_test)

print("5. Avaliando o desempenho do modelo...")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("Relatório de Classificação Completo:")
print(classification_report(y_test, y_pred))

prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
print("-" * 30)
print(f"Métricas da Classe Minoritária (Classe 1):")
print(f"Precisão: {prec:.2f}")
print(f"Recall: {rec:.2f}")

def prever_probabilidades_ensemble(X_data):
    todas_probs = np.array([model.predict_proba(X_data)[:, 1] for model in modelos])
    return np.mean(todas_probs, axis=0)

y_probs = prever_probabilidades_ensemble(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc_score = auc(recall, precision)
print(f"PR AUC: {pr_auc_score:.4f}")
print("Avaliação concluída.")

