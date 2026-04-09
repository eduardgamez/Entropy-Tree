import pandas as pd
import numpy as np

def sort_attributes(df):
    """Ordena los atributos de un DataFrame por su entropía condicional respecto al Target."""
    entropy_var_list = np.array([])
    attributes_list = df.columns[:-1]
    target = df.columns[-1] # Identifica 'Target_Amigo' autónomamente

    for var in attributes_list:
        cond_entropy = 0
        # Dividimos el dataset por cada valor posible del atributo
        for val, subset in df.groupby(var):
            # Calculamos la entropía del TARGET en este subconjunto
            probs = (subset[target].value_counts() / len(subset)).to_numpy()
            ent_subset = -np.sum(probs * np.log2(probs, where=(probs > 0), out=np.zeros_like(probs)))
            
            # Sumamos la entropía pesada
            weight = len(subset) / len(df)
            cond_entropy += weight * ent_subset
            
        entropy_var_list = np.append(entropy_var_list, cond_entropy)

    # Ordenamos: el de menor entropía condicional es el que más información aporta
    sorted_indices = np.argsort(entropy_var_list)
    return np.array(attributes_list)[sorted_indices]

def build_tree(df, sorted_attributes):
    """Construye el árbol de decisión en memoria (como diccionarios anidados) para una velocidad extrema."""
    target = df.columns[-1]
    
    # Caso base 1: Nodo puro (todos los casos son de la misma clase, entropía = 0)
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
        
    # Caso base 2: Si por algún motivo nos quedamos sin atributos para mirar, devolvemos la moda estadística
    if len(sorted_attributes) == 0:
        return df[target].mode()[0]
        
    # Tomamos el primer atributo de la lista de prioridad
    attr = sorted_attributes[0]
    tree = {attr: {}}
    
    # Construimos recursivamente las ramas para cada valor posible en los datos
    for val, subset in df.groupby(attr):
        tree[attr][val] = build_tree(subset, sorted_attributes[1:])
        
    # Guardamos la moda local por si nos llega un caso de prueba con un valor raro no visto en el entrenamiento
    tree[attr]['__moda__'] = df[target].mode()[0]
    
    return tree

def predict_case(tree, row):
    """Clasifica una sola fila de forma ultrarrápida usando el árbol preconstruido."""
    # Si no es un diccionario, es que hemos llegado a una hoja de respuesta (0 o 1)
    if not isinstance(tree, dict):
        return tree
        
    # Extraemos el atributo que toca mirar en este nodo
    attr = list(tree.keys())[0]
    val = row[attr]
    
    # Buscamos la rama que corresponde al valor de la fila
    branch = tree[attr].get(val)
    
    # Si el valor no existía en el entrenamiento, devolvemos la moda para salvar el error
    if branch is None:
        return tree[attr]['__moda__']
        
    # Repetimos para el siguiente nivel navegando por la rama
    return predict_case(branch, row)
