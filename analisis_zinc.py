from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 1. CONFIGURACIÓN DE RUTAS
# =========================================================

ARCHIVO_DATOS = Path("zinc.xlsx")
CARPETA_GRAFICAS = Path("graficas")
CARPETA_RESULTADOS = Path("resultados")

CARPETA_GRAFICAS.mkdir(exist_ok=True)
CARPETA_RESULTADOS.mkdir(exist_ok=True)


# =========================================================
# 2. LECTURA Y PREPARACIÓN DE LOS DATOS
# =========================================================

if not ARCHIVO_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo '{ARCHIVO_DATOS}'. "
        "Coloca zinc.xlsx en la misma carpeta que analisis_zinc.py."
    )

df = pd.read_excel(ARCHIVO_DATOS)

print("Columnas encontradas:", df.columns.tolist())
print("Dimensiones:", df.shape)

columnas_requeridas = {"Fecha", "Precio"}

if not columnas_requeridas.issubset(df.columns):
    raise ValueError(
        "El archivo debe contener columnas llamadas exactamente "
        "'Fecha' y 'Precio'."
    )

df["Fecha"] = pd.to_datetime(
    df["Fecha"],
    errors="coerce",
    dayfirst=True
)

df["Precio"] = pd.to_numeric(
    df["Precio"],
    errors="coerce"
)

df = df.dropna(subset=["Fecha", "Precio"])
df = df.drop_duplicates(subset=["Fecha"])
df = df.sort_values("Fecha").set_index("Fecha")

serie = df["Precio"]

print("\nPrimeras observaciones:")
print(serie.head())

print("\nÚltimas observaciones:")
print(serie.tail())

print("\nNúmero de observaciones:", len(serie))

if len(serie) < 24:
    raise ValueError(
        "Se necesitan al menos 24 observaciones mensuales "
        "para aplicar STL con periodo 12."
    )


# =========================================================
# 3. VERIFICAR SI FALTAN MESES
# =========================================================

fechas_esperadas = pd.date_range(
    start=serie.index.min(),
    end=serie.index.max(),
    freq="MS"
)

fechas_faltantes = fechas_esperadas.difference(serie.index)

if len(fechas_faltantes) > 0:
    print("\nAdvertencia: se detectaron meses faltantes:")
    for fecha in fechas_faltantes:
        print("-", fecha.strftime("%Y-%m"))
else:
    print("\nNo se detectaron meses faltantes.")


# =========================================================
# 4. GRÁFICA DE LA SERIE ORIGINAL
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    serie.index,
    serie
)

plt.title("Precio mensual del zinc")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.grid(alpha=0.3)

plt.savefig(
    CARPETA_GRAFICAS / "serie_original.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 5. DESCOMPOSICIÓN STL
# =========================================================

resultado_stl = STL(
    serie,
    period=12,
    robust=True
).fit()

df["Tendencia"] = resultado_stl.trend
df["Estacionalidad"] = resultado_stl.seasonal
df["Residuo"] = resultado_stl.resid


fuerza_estacional = max(
    0,
    1
    - np.var(resultado_stl.resid)
    / np.var(
        resultado_stl.resid
        + resultado_stl.seasonal
    )
)


fuerza_tendencia = max(
    0,
    1
    - np.var(resultado_stl.resid)
    / np.var(
        resultado_stl.resid
        + resultado_stl.trend
    )
)


print(
    "\nFuerza estacional:",
    round(fuerza_estacional, 3)
)

print(
    "Fuerza de tendencia:",
    round(fuerza_tendencia, 3)
)


fig = resultado_stl.plot()

fig.set_size_inches(
    10,
    8
)

fig.suptitle(
    "Descomposición STL del precio mensual del zinc",
    y=1.02
)

plt.savefig(
    CARPETA_GRAFICAS / "descomposicion_stl.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 6. FUNCIÓN PARA PRUEBA ADF
# =========================================================

def prueba_adf(
    serie_analizada: pd.Series,
    nombre: str
) -> dict:

    resultado = adfuller(
        serie_analizada.dropna(),
        autolag="AIC"
    )

    conclusion = (
        "Estacionaria"
        if resultado[1] < 0.05
        else "No estacionaria"
    )

    print(
        f"\nPRUEBA ADF: {nombre}"
    )

    print("-" * 45)

    print(
        "Estadístico ADF:",
        round(resultado[0], 4)
    )

    print(
        "Valor p:",
        round(resultado[1], 4)
    )

    print(
        "Rezagos utilizados:",
        resultado[2]
    )

    print(
        "Número de observaciones:",
        resultado[3]
    )

    print("Valores críticos:")

    for nivel, valor in resultado[4].items():
        print(
            f"  {nivel}: {valor:.4f}"
        )

    print(
        "Conclusión:",
        conclusion
    )

    return {
        "Serie": nombre,
        "Estadístico ADF": resultado[0],
        "Valor p": resultado[1],
        "Rezagos utilizados": resultado[2],
        "Observaciones": resultado[3],
        "Valor crítico 1%": resultado[4]["1%"],
        "Valor crítico 5%": resultado[4]["5%"],
        "Valor crítico 10%": resultado[4]["10%"],
        "Conclusión": conclusion
    }


# =========================================================
# 7. PRUEBA ADF ORIGINAL Y PRIMERA DIFERENCIA
# =========================================================

resultados_adf = []


resultados_adf.append(
    prueba_adf(
        serie,
        "Precio original"
    )
)


primera_diferencia = serie.diff().dropna()

df["Primera diferencia"] = serie.diff()


resultados_adf.append(
    prueba_adf(
        primera_diferencia,
        "Primera diferencia"
    )
)


pd.DataFrame(
    resultados_adf
).to_excel(
    CARPETA_RESULTADOS
    / "resultados_prueba_adf.xlsx",
    index=False
)


# =========================================================
# 8. GRÁFICA DE LA PRIMERA DIFERENCIA
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    primera_diferencia.index,
    primera_diferencia
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Primera diferencia del precio del zinc"
)

plt.xlabel("Fecha")
plt.ylabel("Cambio mensual")
plt.grid(alpha=0.3)

plt.savefig(
    CARPETA_GRAFICAS
    / "primera_diferencia.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 9. ACF Y PACF DE LA SERIE ORIGINAL
# =========================================================

numero_rezagos = min(
    36,
    len(serie) // 2 - 1
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

plot_acf(
    serie,
    lags=numero_rezagos,
    ax=ax,
    zero=False
)

ax.set_title(
    "ACF de la serie original del zinc"
)

plt.savefig(
    CARPETA_GRAFICAS
    / "acf_original.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


fig, ax = plt.subplots(
    figsize=(10, 5)
)

plot_pacf(
    serie,
    lags=numero_rezagos,
    ax=ax,
    method="ywm",
    zero=False
)

ax.set_title(
    "PACF de la serie original del zinc"
)

plt.savefig(
    CARPETA_GRAFICAS
    / "pacf_original.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 10. ACF Y PACF DE LA PRIMERA DIFERENCIA
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

plot_acf(
    primera_diferencia,
    lags=numero_rezagos,
    ax=ax,
    zero=False
)

ax.set_title(
    "ACF de la primera diferencia del zinc"
)

plt.savefig(
    CARPETA_GRAFICAS
    / "acf_primera_diferencia.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


fig, ax = plt.subplots(
    figsize=(10, 5)
)

plot_pacf(
    primera_diferencia,
    lags=numero_rezagos,
    ax=ax,
    method="ywm",
    zero=False
)

ax.set_title(
    "PACF de la primera diferencia del zinc"
)

plt.savefig(
    CARPETA_GRAFICAS
    / "pacf_primera_diferencia.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 11. DIVISIÓN ENTRENAMIENTO / PRUEBA
# =========================================================

# Con 120 observaciones usamos los últimos
# 24 meses como periodo de prueba.

meses_prueba = (
    24
    if len(serie) >= 60
    else 12
)


entrenamiento = serie.iloc[
    :-meses_prueba
]

prueba = serie.iloc[
    -meses_prueba:
]


print(
    "\nPERIODO DE ENTRENAMIENTO"
)

print(
    entrenamiento.index.min(),
    "a",
    entrenamiento.index.max()
)


print(
    "\nPERIODO DE PRUEBA"
)

print(
    prueba.index.min(),
    "a",
    prueba.index.max()
)


# =========================================================
# 12. FUNCIÓN PARA CALCULAR MÉTRICAS
# =========================================================

def calcular_metricas(
    valores_reales,
    pronostico
):

    valores_reales = np.asarray(
        valores_reales
    )

    pronostico = np.asarray(
        pronostico
    )


    mae = mean_absolute_error(
        valores_reales,
        pronostico
    )


    rmse = np.sqrt(
        mean_squared_error(
            valores_reales,
            pronostico
        )
    )


    mape = np.mean(
        np.abs(
            (
                valores_reales
                - pronostico
            )
            / valores_reales
        )
    ) * 100


    return (
        mae,
        rmse,
        mape
    )


# =========================================================
# 13. MODELOS ARIMA A EVALUAR
# =========================================================

modelos_arima = {

    "ARIMA(0,1,0)": (
        0,
        1,
        0
    ),

    "ARIMA(1,1,0)": (
        1,
        1,
        0
    ),

    "ARIMA(0,1,1)": (
        0,
        1,
        1
    ),

    "ARIMA(1,1,1)": (
        1,
        1,
        1
    )
}


resultados_modelos = []

pronosticos_modelos = {}


# =========================================================
# 14. AJUSTE DE MODELOS ARIMA
# =========================================================

for nombre, orden in modelos_arima.items():

    try:

        modelo = ARIMA(
            entrenamiento,
            order=orden,

            # En zinc no añadimos
            # tendencia determinística explícita.
            trend="n"
        )


        ajuste = modelo.fit()


        pronostico = ajuste.forecast(
            steps=len(prueba)
        )


        pronostico.index = prueba.index


        mae, rmse, mape = calcular_metricas(
            prueba,
            pronostico
        )


        resultados_modelos.append({

            "Modelo": nombre,

            "AIC": ajuste.aic,

            "BIC": ajuste.bic,

            "MAE": mae,

            "RMSE": rmse,

            "MAPE (%)": mape

        })


        pronosticos_modelos[
            nombre
        ] = pronostico


    except Exception as error:

        print(
            f"No se pudo ajustar {nombre}: "
            f"{error}"
        )


# =========================================================
# 15. HOLT CON TENDENCIA AMORTIGUADA
# =========================================================

try:

    modelo_holt = Holt(
        entrenamiento,
        damped_trend=True,
        initialization_method="estimated"
    )


    ajuste_holt = modelo_holt.fit(
        optimized=True
    )


    pronostico_holt = (
        ajuste_holt.forecast(
            len(prueba)
        )
    )


    pronostico_holt.index = (
        prueba.index
    )


    mae_holt, rmse_holt, mape_holt = (
        calcular_metricas(
            prueba,
            pronostico_holt
        )
    )


    resultados_modelos.append({

        "Modelo":
            "Holt con tendencia amortiguada",

        "AIC":
            ajuste_holt.aic,

        "BIC":
            ajuste_holt.bic,

        "MAE":
            mae_holt,

        "RMSE":
            rmse_holt,

        "MAPE (%)":
            mape_holt

    })


    pronosticos_modelos[
        "Holt con tendencia amortiguada"
    ] = pronostico_holt


except Exception as error:

    print(
        "No se pudo ajustar Holt "
        "con tendencia amortiguada: "
        f"{error}"
    )


# =========================================================
# 16. TABLA COMPARATIVA DE MODELOS
# =========================================================

comparacion = pd.DataFrame(
    resultados_modelos
)


if comparacion.empty:

    raise RuntimeError(
        "No fue posible ajustar "
        "ninguno de los modelos."
    )


comparacion = (
    comparacion
    .sort_values(
        by="RMSE"
    )
    .reset_index(
        drop=True
    )
)


comparacion = comparacion.round({

    "AIC": 2,

    "BIC": 2,

    "MAE": 4,

    "RMSE": 4,

    "MAPE (%)": 2

})


print(
    "\nCOMPARACIÓN DE MODELOS"
)


print(
    comparacion.to_string(
        index=False
    )
)


comparacion.to_excel(
    CARPETA_RESULTADOS
    / "comparacion_modelos_zinc.xlsx",
    index=False
)


# =========================================================
# 17. SELECCIÓN DEL MODELO CON MENOR RMSE
# =========================================================

mejor_modelo = (
    comparacion.iloc[0]
)


nombre_mejor_modelo = (
    mejor_modelo["Modelo"]
)


print(
    "\nMODELO CON MENOR RMSE"
)


print(
    "Modelo:",
    nombre_mejor_modelo
)


print(
    "MAE:",
    mejor_modelo["MAE"]
)


print(
    "RMSE:",
    mejor_modelo["RMSE"]
)


print(
    "MAPE:",
    mejor_modelo["MAPE (%)"],
    "%"
)


# =========================================================
# 18. GRÁFICA DE COMPARACIÓN DE PRONÓSTICOS
# =========================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    entrenamiento.index,
    entrenamiento,
    label="Entrenamiento"
)


plt.plot(
    prueba.index,
    prueba,
    label="Valores reales",
    linewidth=2
)


for nombre, pronostico in (
    pronosticos_modelos.items()
):

    plt.plot(
        pronostico.index,
        pronostico,
        label=nombre
    )


plt.title(
    "Comparación de pronósticos del precio del zinc"
)

plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.grid(alpha=0.3)


plt.savefig(
    CARPETA_GRAFICAS
    / "comparacion_pronosticos.png",
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# 19. GRÁFICA DE VALIDACIÓN DEL MEJOR MODELO
# =========================================================

pronostico_mejor = (
    pronosticos_modelos[
        nombre_mejor_modelo
    ]
)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    entrenamiento.index,
    entrenamiento,
    label="Entrenamiento"
)


plt.plot(
    prueba.index,
    prueba,
    label="Valores reales"
)


plt.plot(
    pronostico_mejor.index,
    pronostico_mejor,
    label=(
        f"Pronóstico - "
        f"{nombre_mejor_modelo}"
    ),
    linewidth=2
)


plt.title(
    "Validación del modelo seleccionado: "
    f"{nombre_mejor_modelo}"
)

plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.grid(alpha=0.3)


plt.savefig(
    CARPETA_GRAFICAS
    / "mejor_modelo_validacion.png",
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# 20. AJUSTE FINAL CON TODA LA SERIE
# =========================================================

# Después de seleccionar el mejor modelo utilizando
# 2023-2024 como periodo de prueba, se vuelve a
# ajustar utilizando las 120 observaciones.


if nombre_mejor_modelo.startswith(
    "ARIMA"
):

    orden_texto = (
        nombre_mejor_modelo
        .replace(
            "ARIMA(",
            ""
        )
        .replace(
            ")",
            ""
        )
    )


    orden_mejor = tuple(

        int(valor)

        for valor
        in orden_texto.split(",")

    )


    modelo_final = ARIMA(
        serie,
        order=orden_mejor,
        trend="n"
    )


    ajuste_final = (
        modelo_final.fit()
    )


    pronostico_2025 = (
        ajuste_final.forecast(
            steps=12
        )
    )


else:

    modelo_final = Holt(
        serie,
        damped_trend=True,
        initialization_method="estimated"
    )


    ajuste_final = (
        modelo_final.fit(
            optimized=True
        )
    )


    pronostico_2025 = (
        ajuste_final.forecast(
            12
        )
    )


# =========================================================
# 21. FECHAS DEL PRONÓSTICO 2025
# =========================================================

fechas_2025 = pd.date_range(

    start=(
        serie.index.max()
        + pd.offsets.MonthBegin(1)
    ),

    periods=12,

    freq="MS"

)


pronostico_2025.index = (
    fechas_2025
)


df_pronostico_2025 = pd.DataFrame({

    "Fecha":
        fechas_2025,

    "Precio pronosticado":
        pronostico_2025.values

})


df_pronostico_2025.to_excel(

    CARPETA_RESULTADOS
    / "pronostico_zinc_2025.xlsx",

    index=False
)


print(
    "\nPRONÓSTICO DEL PRECIO "
    "DEL ZINC PARA 2025"
)


print(
    df_pronostico_2025.to_string(
        index=False
    )
)


# =========================================================
# 22. GRÁFICA DEL PRONÓSTICO 2025
# =========================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    serie.index,
    serie,
    label="Datos históricos"
)


plt.plot(
    pronostico_2025.index,
    pronostico_2025,
    label=(
        f"Pronóstico 2025 - "
        f"{nombre_mejor_modelo}"
    ),
    linewidth=2
)


plt.axvline(
    serie.index.max(),
    linestyle="--",
    alpha=0.7
)


plt.title(
    "Pronóstico del precio mensual "
    "del zinc para 2025"
)


plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.grid(alpha=0.3)


plt.savefig(
    CARPETA_GRAFICAS
    / "pronostico_zinc_2025.png",
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# 23. EXPORTAR ANÁLISIS COMPLETO
# =========================================================

df.to_excel(
    CARPETA_RESULTADOS
    / "zinc_analisis_completo.xlsx"
)


# =========================================================
# 24. RESUMEN DEL ANÁLISIS
# =========================================================

resumen = pd.DataFrame({

    "Indicador": [

        "Número de observaciones",

        "Fuerza estacional",

        "Fuerza de tendencia",

        "Periodo de prueba",

        "Modelo con menor RMSE",

        "MAE del modelo seleccionado",

        "RMSE del modelo seleccionado",

        "MAPE del modelo seleccionado"

    ],


    "Resultado": [

        len(serie),

        round(
            fuerza_estacional,
            3
        ),

        round(
            fuerza_tendencia,
            3
        ),

        meses_prueba,

        nombre_mejor_modelo,

        mejor_modelo["MAE"],

        mejor_modelo["RMSE"],

        mejor_modelo["MAPE (%)"]

    ]

})


resumen.to_excel(

    CARPETA_RESULTADOS
    / "resumen_analisis_zinc.xlsx",

    index=False
)


# =========================================================
# 25. MENSAJE FINAL
# =========================================================

print(
    "\nAnálisis terminado."
)


print(
    "\nArchivos generados "
    "en graficas/:"
)


print(
    "- serie_original.png"
)

print(
    "- primera_diferencia.png"
)

print(
    "- descomposicion_stl.png"
)

print(
    "- acf_original.png"
)

print(
    "- pacf_original.png"
)

print(
    "- acf_primera_diferencia.png"
)

print(
    "- pacf_primera_diferencia.png"
)

print(
    "- comparacion_pronosticos.png"
)

print(
    "- mejor_modelo_validacion.png"
)

print(
    "- pronostico_zinc_2025.png"
)


print(
    "\nArchivos generados "
    "en resultados/:"
)


print(
    "- zinc_analisis_completo.xlsx"
)

print(
    "- resultados_prueba_adf.xlsx"
)

print(
    "- comparacion_modelos_zinc.xlsx"
)

print(
    "- resumen_analisis_zinc.xlsx"
)

print(
    "- pronostico_zinc_2025.xlsx"
)
