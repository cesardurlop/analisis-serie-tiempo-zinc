# Análisis de serie de tiempo del precio del zinc

Este repositorio contiene el análisis de la serie de tiempo mensual del precio del zinc correspondiente al periodo de enero de 2015 a diciembre de 2024.

El objetivo es estudiar el comportamiento temporal de la serie, identificar tendencia, estacionalidad y estacionariedad, y comparar distintos modelos de pronóstico para estimar los precios mensuales de 2025.

## Datos

La base de datos contiene 120 observaciones mensuales correspondientes al periodo:

- Inicio: enero de 2015
- Fin: diciembre de 2024
- Frecuencia: mensual
- Variable analizada: precio del zinc

El archivo utilizado es:

`zinc.xlsx`

El archivo debe contener exactamente las siguientes columnas:

- `Fecha`
- `Precio`

## Metodología

El análisis se realiza mediante las siguientes etapas:

1. Lectura y preparación de los datos.
2. Verificación de meses faltantes.
3. Visualización de la serie original.
4. Descomposición STL con periodo de 12 meses.
5. Cálculo de la fuerza de tendencia y estacionalidad.
6. Prueba de Dickey-Fuller aumentada (ADF) sobre la serie original.
7. Aplicación de la primera diferencia.
8. Prueba ADF sobre la serie diferenciada.
9. Análisis de las funciones ACF y PACF.
10. División de la serie en entrenamiento y prueba.
11. Comparación de modelos de pronóstico.
12. Evaluación mediante MAE, RMSE y MAPE.
13. Selección del modelo con mejor desempeño predictivo.
14. Ajuste final del modelo seleccionado utilizando toda la serie.
15. Pronóstico mensual para enero-diciembre de 2025.

## Modelos evaluados

Se comparan los siguientes modelos:

- ARIMA(0,1,0)
- ARIMA(1,1,0)
- ARIMA(0,1,1)
- ARIMA(1,1,1)
- Holt con tendencia amortiguada

Los modelos ARIMA se evalúan debido a que la serie original no presenta estacionariedad y la primera diferencia permite estabilizar su comportamiento.

El modelo ARIMA(0,1,0) se incluye como referencia o modelo base, mientras que los demás modelos permiten evaluar si componentes autorregresivos o de media móvil mejoran la capacidad predictiva.

El método de Holt con tendencia amortiguada se incorpora como un modelo externo de comparación.

## Estacionalidad

La descomposición STL permite evaluar la presencia de patrones estacionales anuales.

En el caso del zinc, la fuerza estacional observada es baja, por lo que inicialmente no se consideran modelos SARIMA con componentes estacionales.

## Validación

Para evaluar la capacidad predictiva de los modelos se utilizan los últimos 24 meses de la serie como conjunto de prueba.

De esta forma:

- Entrenamiento: enero de 2015 a diciembre de 2022.
- Prueba: enero de 2023 a diciembre de 2024.

Los modelos se comparan mediante las siguientes métricas:

- MAE: Error Absoluto Medio.
- RMSE: Raíz del Error Cuadrático Medio.
- MAPE: Error Porcentual Absoluto Medio.

Para los modelos ARIMA también se reportan:

- AIC
- BIC

La selección principal se realiza considerando el desempeño fuera de muestra, especialmente el RMSE, acompañado por MAE y MAPE.

## Pronóstico 2025

Después de seleccionar el modelo con mejor desempeño durante el periodo de prueba, el modelo se vuelve a estimar utilizando las 120 observaciones disponibles entre enero de 2015 y diciembre de 2024.

Posteriormente se generan pronósticos mensuales para los 12 meses de 2025.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── analisis_zinc.py
├── zinc.xlsx
├── graficas/
└── resultados/
