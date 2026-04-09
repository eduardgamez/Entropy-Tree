import sys
import os
import pandas as pd
import numpy as np

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.functions import sort_attributes, build_tree, predict_case

# 1. Carga de datos
train, test = pd.read_csv('dataset/dataset.csv'), pd.read_csv('test/test_cases.csv')
target = train.columns[-1]

# 2. Entrenamos el árbol
tree = build_tree(train, sort_attributes(train))

# 3. Clasificación
preds_tree = test.apply(lambda r: predict_case(tree, r), axis=1)
preds_iff = test['Respuesta_IFF'].apply(lambda x: 1 if x == 'Valida' else 0)

# 4. Cálculo de Accuracy
acc_tree = (preds_tree == test[target]).mean()
acc_iff = (preds_iff == test[target]).mean()

print(f"Accuracy SOLO IFF: {acc_iff:.2%}")
print(f"Accuracy ÁRBOL ID3: {acc_tree:.2%}")
print(f"Mejora del árbol: {acc_tree - acc_iff:.2%}")
