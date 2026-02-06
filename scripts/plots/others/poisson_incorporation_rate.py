import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Parámetros de la simulación
longitud_puente = 200  # metros
velocidad_media = 1.4  # m/s
velocidad_std = 0.2
prom_peatones_en_puente = 100
tiempo_simulacion = 2000  # segundos
n_simulaciones = 10  # simulaciones de fondo

def simular_peatones(longitud, v_media, v_std, n_objetivo, tiempo_total):
    tiempo_cruce_prom = longitud / v_media
    lambda_llegadas = n_objetivo / tiempo_cruce_prom
    tiempo_entre_llegadas = 1 / lambda_llegadas

    tiempos_llegada = []
    t = 0
    while t < tiempo_total:
        intervalo = np.random.exponential(tiempo_entre_llegadas)
        t += intervalo
        if t < tiempo_total:
            tiempos_llegada.append(t)
    tiempos_llegada = np.array(tiempos_llegada)

    velocidades = np.random.normal(v_media, v_std, size=len(tiempos_llegada))
    velocidades = np.clip(velocidades, 0.5, None)
    tiempos_cruce = longitud / velocidades
    tiempos_salida = tiempos_llegada + tiempos_cruce

    eventos = np.concatenate([
        np.column_stack((tiempos_llegada, np.ones_like(tiempos_llegada))),
        np.column_stack((tiempos_salida, -np.ones_like(tiempos_salida)))
    ])
    eventos = eventos[np.argsort(eventos[:, 0])]
    eventos = eventos[eventos[:, 0] <= tiempo_total]

    tiempos = [0]
    conteo = [0]
    actual = 0
    for tiempo, cambio in eventos:
        tiempos.append(tiempo)
        actual += cambio
        conteo.append(actual)

    if tiempos[-1] < tiempo_total:
        tiempos.append(tiempo_total)
        conteo.append(actual)

    return np.array(tiempos), np.array(conteo)


# Guardar resultados
import os
import scipy.io as sio

# Configuración de rutas
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
output_base = os.path.join(repo_root, "assets", "figures")
output_png = os.path.join(output_base, "png")
output_pdf = os.path.join(output_base, "pdf")
output_data = os.path.join(script_dir, "poisson_data.mat")

os.makedirs(output_png, exist_ok=True)
os.makedirs(output_pdf, exist_ok=True)

# Recalcular las simulaciones para guardar los datos
t_back_list = []
c_back_list = []

# Crear la figura
plt.figure(figsize=(14, 6))

# Dibujar 10 simulaciones grises de fondo
for _ in range(n_simulaciones):
    t_sim, c_sim = simular_peatones(longitud_puente, velocidad_media, velocidad_std,
                                    prom_peatones_en_puente, tiempo_simulacion)
    plt.step(t_sim, c_sim, where='post', color='gray', linewidth=1, alpha=0.4)
    t_back_list.append(t_sim)
    c_back_list.append(c_sim)

# Simulación principal en negro
t_principal, c_principal = simular_peatones(longitud_puente, velocidad_media, velocidad_std,
                                            prom_peatones_en_puente, tiempo_simulacion)
plt.step(t_principal, c_principal, where='post', color='black', linewidth=2.5, label='Single Simulation')

# Línea roja horizontal: promedio objetivo
plt.axhline(y=prom_peatones_en_puente, color='red', linestyle='--', linewidth=2, label='Target Crowd Density')

# Personalización del gráfico
plt.xlabel("Time [s]")
plt.ylabel("Number of pedestrians")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar figuras
filename = "poisson_incorporation_rate"
plt.savefig(os.path.join(output_png, f"{filename}.png"), dpi=300)
plt.savefig(os.path.join(output_pdf, f"{filename}.pdf"))
print(f"Figuras guardadas en {output_base}")

# Guardar datos para MATLAB (para generar el .fig)
mat_data = {
    't_back': np.array(t_back_list, dtype=object),
    'c_back': np.array(c_back_list, dtype=object),
    't_main': t_principal,
    'c_main': c_principal,
    'target_val': prom_peatones_en_puente
}
sio.savemat(output_data, mat_data)
print(f"Datos guardados en {output_data} para generación de .fig en MATLAB")

plt.show()
