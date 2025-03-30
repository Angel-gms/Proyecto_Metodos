import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm, t
import streamlit as st

# ================================
# Inciso (a): Descarga y descripción del activo
# ================================
st.title("Proyecto MCF - Análisis de Activo Financiero")

# Selecciona el activo: En este ejemplo usamos AAPL (Apple Inc.)
asset = "AAPL"
st.subheader("Descripción del Activo")
st.write(
    "El activo seleccionado es Apple Inc. (AAPL), una empresa líder en tecnología y electrónica de consumo. "
    "Se ha descargado la información financiera desde el año 2010 usando Yahoo Finance."
)

# Descarga de datos desde 2010
data = yf.download(asset, start="2010-01-01")['Adj Close']
data = data.dropna()
st.write("Muestra de datos descargados:")
st.dataframe(data.tail())

# ================================
# Inciso (b): Cálculo de rendimientos diarios y estadísticas
# ================================
returns = data.pct_change().dropna()  # cálculo de retornos diarios
mean_ret = returns.mean()
skew_ret = skew(returns)
excess_kurt = kurtosis(returns)  # Por defecto, devuelve la curtosis de Fisher (exceso de curtosis)

st.subheader("Estadísticas de Rendimientos Diarios")
st.write(f"**Media:** {mean_ret:.6f}")
st.write(f"**Sesgo:** {skew_ret:.6f}")
st.write(f"**Exceso de Curtosis:** {excess_kurt:.6f}")

# ================================
# Inciso (c): Cálculo de VaR y Expected Shortfall (ES)
# Se realizan cuatro aproximaciones:
#   1. Paramétrico normal.
#   2. Paramétrico t-Student.
#   3. Histórico.
#   4. Monte Carlo (suponiendo distribución normal).
# ================================
alpha_levels = [0.95, 0.975, 0.99]
results = {"Método": [], "α": [], "VaR": [], "ES": []}

# 1. Método Paramétrico con Distribución Normal
mu = returns.mean()
sigma = returns.std()
for alpha in alpha_levels:
    # Se utiliza el cuantil de la cola inferior (1 - α)
    z = norm.ppf(1 - alpha)
    # Se define VaR como la pérdida que excede con probabilidad (1-α)
    var_normal = -(mu + sigma * z)
    # Fórmula para el Expected Shortfall (ES) bajo normal:
    es_normal = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    results["Método"].append("Paramétrico Normal")
    results["α"].append(alpha)
    results["VaR"].append(var_normal)
    results["ES"].append(es_normal)

# 2. Método Paramétrico con Distribución t-Student
# Se ajustan los parámetros de la t-Student a los datos de retornos
df, loc, scale = t.fit(returns)
for alpha in alpha_levels:
    t_quantile = t.ppf(1 - alpha, df, loc=loc, scale=scale)
    var_t = -t_quantile
    # Para el ES bajo t-Student se realiza una aproximación por simulación:
    sim = t.rvs(df, loc=loc, scale=scale, size=100000)
    threshold = np.percentile(sim, (1 - alpha) * 100)
    es_t = -sim[sim <= threshold].mean()
    results["Método"].append("Paramétrico t-Student")
    results["α"].append(alpha)
    results["VaR"].append(var_t)
    results["ES"].append(es_t)

# 3. Método Histórico
for alpha in alpha_levels:
    var_hist = -np.percentile(returns, (1 - alpha) * 100)
    threshold_hist = np.percentile(returns, (1 - alpha) * 100)
    es_hist = -returns[returns <= threshold_hist].mean()
    results["Método"].append("Histórico")
    results["α"].append(alpha)
    results["VaR"].append(var_hist)
    results["ES"].append(es_hist)

# 4. Método Monte Carlo (suponiendo distribución normal)
np.random.seed(42)  # para reproducibilidad
simulations = 100000
simulated_returns = np.random.normal(mu, sigma, simulations)
for alpha in alpha_levels:
    var_mc = -np.percentile(simulated_returns, (1 - alpha) * 100)
    threshold_mc = np.percentile(simulated_returns, (1 - alpha) * 100)
    es_mc = -simulated_returns[simulated_returns <= threshold_mc].mean()
    results["Método"].append("Monte Carlo")
    results["α"].append(alpha)
    results["VaR"].append(var_mc)
    results["ES"].append(es_mc)

results_df = pd.DataFrame(results)
st.subheader("Resultados de VaR y Expected Shortfall (ES)")
st.table(results_df)
