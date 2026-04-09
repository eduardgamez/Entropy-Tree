import sys
import os
import pandas as pd

# Añadimos el directorio raíz al path para poder importar algorithms de scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.functions import sort_attributes, build_tree, predict_case

def main():
    print("1. Cargando datasets...")
    train_df = pd.read_csv('dataset/dataset.csv')
    test_df = pd.read_csv('test/test_cases.csv')
    
    print("2. Calculando entropía y ordenando atributos...")
    sorted_attrs = sort_attributes(train_df)
    print(f"Orden: {sorted_attrs}")
    
    print("3. Construyendo el árbol en memoria (fase de entrenamiento)...")
    tree = build_tree(train_df, sorted_attrs)
    
    print("4. Evaluando casos de prueba (fase de ejecución ultrarrápida)...")
    correctos = 0
    total = len(test_df)
    target = train_df.columns[-1]
    
    # Separamos los atributos de la respuesta real
    features_test = test_df.drop(columns=[target])
    reales = test_df[target].to_numpy()
    
    # Aquí es donde usamos predict_case de forma masiva pero caso a caso
    for i, row in features_test.iterrows():
        prediccion = predict_case(tree, row)
        if prediccion == reales[i]:
            correctos += 1
            
    precision = (correctos / total) * 100
    print(f"\nResultados: {correctos}/{total} aciertos ({precision:.2f}%)")

if __name__ == "__main__":
    main()
