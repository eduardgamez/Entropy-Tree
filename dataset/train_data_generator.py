import pandas as pd
import numpy as np
import random
import os

def generate_tactical_dataset(n_samples=10000):
    data = []
    
    # Definición de tipos de objetos y sus perfiles lógicos
    # Formato: (Nombre, Es_Amigo[1/0], Prob_IFF_Valido, RCS_esperado)
    profiles = [
        ('Caza_Aliado', 1, 0.95, 'Medio'),
        ('Caza_Aliado_Stealth', 1, 0.90, 'Pequeño'),
        ('Comercial', 1, 0.99, 'Grande'),
        ('Dron_Aliado', 1, 0.85, 'Pequeño'),
        ('Caza_Enemigo', 0, 0.05, 'Medio'),
        ('Caza_Enemigo_Stealth', 0, 0.01, 'Muy_Pequeño'),
        ('Dron_Enemigo', 0, 0.02, 'Pequeño'),
        ('Misil_Crucero', 0, 0.01, 'Pequeño')
    ]

    for _ in range(n_samples):
        # Seleccionamos un perfil aleatorio
        name, is_friend, p_iff, rcs_base = random.choice(profiles)
        
        # 1. Respuesta IFF (Simulamos fallos como el de Kuwait)
        iff_rand = random.random()
        if iff_rand < p_iff:
            iff = 'Valida'
        elif iff_rand < p_iff + 0.05:
            iff = 'Invalida' # Error de código
        else:
            iff = 'Ausente' # No responde o apagado

        # 2. Perfil de Vuelo
        if is_friend:
            perfil = np.random.choice(['Correcto', 'Desviado'], p=[0.8, 0.2])
        else:
            perfil = np.random.choice(['Erratico', 'Desviado'], p=[0.7, 0.3])

        # 3. Firma Radar (RCS) - Con ruido para engañar al árbol
        rcs = rcs_base
        if random.random() < 0.1: # 10% de error de lectura
            rcs = random.choice(['Muy_Pequeño', 'Pequeño', 'Medio', 'Grande'])

        # 4. Proximidad Aliada (en Km)
        if is_friend:
            prox = random.uniform(0, 50)
        else:
            prox = random.uniform(30, 200)
        
        # Discretizamos proximidad para ID3 (Cerca, Media, Lejos)
        if prox < 20: prox_cat = 'Cerca'
        elif prox < 80: prox_cat = 'Media'
        else: prox_cat = 'Lejos'

        data.append([rcs, iff, perfil, prox_cat, is_friend])

    columns = ['Firma_Radar', 'Respuesta_IFF', 'Perfil_Vuelo', 'Proximidad_Aliada', 'Target_Amigo']
    df = pd.DataFrame(data, columns=columns)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generado con {n_samples} registros.")

if __name__ == "__main__":
    generate_tactical_dataset()