# Badillo Santos Laura Berenice
# Martinez Suárez Ángel Gabriel
#######################################################
# Importamos las librerias que utilizamos 
#######################################################

import yfinance as yf 
import pandas as pd 
import numpy as np
import scipy.stats as stats
from scipy.stats import skew, kurtosis, norm, t
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image 

st.title(" " *20 + "Proyecto 1 - Value-At-Risk y Expected Shortfall.")

st.write("Nombres: * Martínez Suárez Ángel Gabriel ---- * Badillo Santos Laura Berenice")

imagen = Image.open('MCF_imagen.jpeg')
st.image(imagen)
#######################################################
# (a)~> Descargamos los datos de nuestro activo
#######################################################

# Seleccionamos para el proyecto BBVA
st.subheader("Descripción del Activo (BBVA)")
st.write(
    "Banco Bilbao Vizcaya Argentaria, S.A. presta servicios de banca minorista, "
    "banca mayorista y gestión de activos principalmente en España, México, Turquía, Sudamérica, "
    "resto de Europa, Estados Unidos y Asia. La empresa ofrece cuentas de ahorro, depósitos a la vista"
    " y depósitos a plazo; y préstamos hipotecarios para viviendas, otros hogares, tarjetas de crédito y empresas "
    "y el sector público, así como financiación al consumo. Ofrece negocios de seguros y gestión de activos, "
    "incluidos corporativos, comerciales, PYMES, sistemas de pago, banca minorista, privada y de inversión,"
    " seguros de pensiones y vida, arrendamiento, factoraje y corretaje. La empresa ofrece sus productos "
    "a través de canales en línea y móviles. Banco Bilbao Vizcaya Argentaria, S.A. fue fundado en 1857 "
    "y tiene su sede en Bilbao, España."
)
# Tomamos los datos desde 2010
datos = yf.download("BBVA", start="2010-01-01")['Close']
ultima_fecha = str(datos.index[-1])
st.write(f"Muestra los datos descargados del precio de cierre hasta {ultima_fecha[:10]}:")
st.dataframe(datos)

#######################################################
# (b)~> Calculo de rendimientos diarios
#######################################################

rendimientos = datos.pct_change().dropna()
rendimiento_md = rendimientos.mean()
sesgo = skew(rendimientos)
kurtosiss = kurtosis(rendimientos) 

st.subheader("Estadísticas de Rendimientos Diarios")

col1, col2, col3= st.columns(3)
col1.metric("Rendimiento Medio Diario", f"{rendimiento_md.iloc[0]:.6%}")
col2.metric("Sesgo", f"{sesgo.item():.6f}")
col3.metric("Kurtosis", f"{kurtosiss.item():.6f}")

# Histograma de rendimientos
plt.style.use('dark_background')
st.subheader("Histograma de Rendimientos")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rendimientos, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(rendimiento_md.iloc[0], color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_md.iloc[0]:.6%}")
ax.legend()
ax.set_xlabel("Rendimiento Diario")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

#######################################################
# (c)~> Calculo de VaR y Expected Shortfall (ES)
#######################################################

st.subheader("Metricas de riesgo VaR y ES")
# Seleccion de alpha
alphas = [0.95, 0.975, 0.99]
seleccionador = st.selectbox("Selecciona una alpha (α)", alphas)

if seleccionador:
    st.write(f"Metricas para un alpha(α) = {seleccionador}")
    media = np.mean(rendimientos)
    des_est = np.std(rendimientos)

    # VaR Parametrico normal
    var_norm = norm.ppf(1-seleccionador,media,des_est)
    # VaR Parametrico t-student
    grad_lib = len(rendimientos) - 1
    var_t = t.ppf(1-seleccionador, grad_lib, media, des_est)
    # VaR historico
    var_h = (rendimientos.quantile(1-seleccionador))
    # VaR Monte Carlo
    n_sim = 100000
    rendimientos_sim = np.random.normal(media, des_est, n_sim)
    var_mc = np.percentile(rendimientos_sim, 100-seleccionador*100)
    # CVaR o ES 
    cvar = rendimientos[rendimientos<=var_h].mean()

    col4, col5, col6, col7, col8= st.columns(5)
    col4.metric(f"VaR parametrico normal", f"{var_norm.item():.3%}")
    col5.metric("VaR parametrico t-student", f"{var_t.item():.3%}")
    col6.metric("VaR Historico", f"{var_h.item():.3%}")
    col7.metric("VaR Monte Carlo", f"{var_mc.item():.3%}")
    col8.metric("CVaR", f"{cvar.item():.3%}")

    #######################################################
    # (d)~> Rolling windows
    #######################################################
    rolling = rendimientos.rolling(window=252)
    rolling_media = rolling.mean()
    rolling_des_est = rolling.std()

    rolling_var_h = rolling.quantile(1-seleccionador)
    rolling_cvar_h = rolling.apply(lambda x: x[x <= x.quantile(1-seleccionador)].mean())
    rolling_var_norm = norm.ppf(1-seleccionador, rolling_media, rolling_des_est)
    rolling_cvar_norm = rolling.apply(lambda x: x[x <= norm.ppf(1 - seleccionador, x.mean(), x.std())].mean())


    vaR_rolling_df = pd.DataFrame({'Date': rendimientos.index,
                                   'Rolling VaR normal': rolling_var_norm.squeeze(),
                                   'Rolling VaR historico':rolling_var_h.squeeze(),
                                   'Rolling CVaR normal': rolling_cvar_norm.squeeze(),
                                   'Rolling CVaR historico': rolling_cvar_h.squeeze()})
    vaR_rolling_df.set_index('Date', inplace=True)

    # Crear una gráfica que muestre la serie de retornos diarios (en %) y los estimados de VaR y ES
    fig, ax = plt.subplots(figsize=(14,7))

    # Graficar los retornos diarios (convertidos a porcentaje)
    ax.plot(rendimientos.index, rendimientos * 100, color='lightblue', alpha=0.5, label="Retornos diarios")

    plt.plot(vaR_rolling_df.index, (vaR_rolling_df['Rolling VaR normal']*100).round(5), label=f'Rolling VaR parametrico normal al {seleccionador}%', color='red')
    plt.plot(vaR_rolling_df.index, (vaR_rolling_df['Rolling CVaR normal']*100).round(5), label=f'Rolling CVaR parametrico normal al {seleccionador}%', color='purple')
    plt.plot(vaR_rolling_df.index, (vaR_rolling_df['Rolling VaR historico']*100).round(5), label=f'Rolling VaR historico al {seleccionador}%', color='blue')
    plt.plot(vaR_rolling_df.index, (vaR_rolling_df['Rolling CVaR historico']*100).round(5), label=f'Rolling CVaR historico al {seleccionador}%', color='orange')

    ax.set_title("Retornos diarios con VaR y ES con ventana móvil de 252 retornos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Porcentaje (%)")
    ax.legend(loc="upper left")
    ax.grid(True)

    st.pyplot(fig)


#######################################################
# (e)~> Violaciones 
#######################################################
# Quitamos lon NA's
rolling_var_h = rolling_var_h.dropna()
rolling_cvar_h = rolling_cvar_h.dropna()

q_95 = norm.ppf(0.05)  
q_99 = norm.ppf(0.01)  

# VaR = -qₐ × σ₂₅₂
rolling_VaR_95 = q_95 * rolling_des_est
rolling_VaR_99 = q_99 * rolling_des_est

# Desplazar las predicciones en 1 día para predecir el rendimiento del día siguiente
var_pred = rolling_var_h.shift(1)
cvar_pred = rolling_cvar_h.shift(1)
rolling_VaR_95_pred = rolling_VaR_95.dropna().shift(1)
rolling_VaR_99_pred = rolling_VaR_99.dropna().shift(1)

# Restricción: Solo se pueden evaluar los días en los que existe una predicción (después de la ventana inicial)
fechas = var_pred.dropna().index

# Evaluar violaciones: se considera violación si el rendimiento real es menor que la predicción
violation_var = (rendimientos.loc[fechas] < var_pred.loc[fechas]).astype(int)
violation_cvar = (rendimientos.loc[fechas] < cvar_pred.loc[fechas]).astype(int)
violations_VaR_95 = (rendimientos.loc[fechas] < rolling_VaR_95_pred.loc[fechas]).astype(int)
violations_VaR_99 = (rendimientos.loc[fechas] < rolling_VaR_99_pred.loc[fechas]).astype(int)

num_total = len(fechas)
num_violations_var = violation_var.sum()
num_violations_cvar = violation_cvar.sum()
num_violations_95 = violations_VaR_95.sum()
num_violations_99 = violations_VaR_99.sum()

resultados_df = pd.DataFrame({
    "Medida": [f"VaR {seleccionador}%", f"CVaR {seleccionador}%"],
    "Número de violaciones": [num_violations_var.iloc[0], num_violations_cvar.iloc[0]],
    "Porcentaje (%)": [
        round(num_violations_var.iloc[0] / num_total * 100, 4),
        round(num_violations_cvar.iloc[0] / num_total * 100, 4)
    ]
})

# Tabla con resultdo de violaciones
st.subheader("Violaciones de VaR historico")
st.table(resultados_df)

#######################################################
# (f)~> forumula
#######################################################
# Crear un DataFrame con las violaciones de rolling window
results_df = pd.DataFrame({
    "Medida": ["VaR 95%", "VaR 99%"],
    "Número de Violaciones": [num_violations_95.iloc[0], num_violations_99.iloc[0]],
    "Porcentaje de Violaciones (%)": [
        round(num_violations_95.iloc[0] / num_total * 100, 4),
        round(num_violations_99.iloc[0] / num_total * 100, 4)
    ]
})

# Gráfica: Se muestran los rendimientos diarios (en porcentaje) y las estimaciones de VaR
fig, ax = plt.subplots(figsize=(14,7))
ax.plot(rendimientos.index, rendimientos * 100, color='gray', alpha=0.5, label="Retornos diarios (%)")
ax.plot(rolling_VaR_95_pred.index, rolling_VaR_95_pred * 100, label="VaR 95%", color='red')
ax.plot(rolling_VaR_99_pred.index, rolling_VaR_99_pred * 100, label="VaR 99%", color='blue')
ax.set_title("Retornos diarios y VaR con Rolling Window (ventana de 252 días)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Porcentaje (%)")
ax.legend(loc="upper left")
ax.grid(True)

st.pyplot(fig)

# Tabla con resultdo de violaciones
st.subheader("Violaciones VaR Rolling Window")
st.table(results_df)

