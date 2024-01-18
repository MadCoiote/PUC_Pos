import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

data = pd.read_csv("AirPassengers.csv", parse_dates=["Month"], index_col="Month")
print(data.head(10))
print(data.describe())

x = np.arange(0,data.shape[0])
y_original = data["#Passengers"].to_numpy()

# plt.plot(data, color="black")
# plt.title("Dados Originais")
# plt.ylabel("# de Passageiros")

# plt.figure()
# resultado1 = seasonal_decompose(data, model="multiplicative")
# resultado1.plot()

# plt.figure()
# resultado2 = seasonal_decompose(data, model="additive")
# resultado2.plot()

#ESTACIONARIEDADE DA SÉRIE ORIGINAL: Teste de Dickey-Fuller
teste = adfuller(data["#Passengers"])
print("Valor p do teste de Dickey-Fuller com os dados originais: {:.2%}.".format(teste[1]))

#DIFERENCIAÇÃO DE PRIMEIRA ORDEM
y_diff = np.diff(data["#Passengers"])
# plt.figure()
# plt.plot(y_diff, color="black")
# plt.title("Dados com Diferenciação de Ordem 1")
# plt.ylabel("# de Passageiros")

teste = adfuller(y_diff)
print("Valor p do teste de Dickey-Fuller com os dados diferenciados de ordem 1: {:.2%}.".format(teste[1]))

#DIFERENCIAÇÃO DE SEGUNDA ORDEM
y_diff2 = np.diff(y_diff)
# plt.figure()
# plt.plot(y_diff2, color="black")
# plt.title("Dados com Diferenciação de Ordem 2")
# plt.ylabel("# de Passageiros")

teste = adfuller(y_diff2)
print("Valor p do teste de Dickey-Fuller com os dados diferenciados de ordem 2: {:.2%}.".format(teste[1]))

#PREVISOES
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2, optimized=False)
fcast1 = fit1.forecast(12).rename(r'\alpha=0.2$')                                                      #previsão 12 meses a frente
# plt.figure()
# fcast1.plot(marker=".", color="blue", legend=True)
# fit1.fittedvalues.plot(marker=".", color="red")

fit2 = Holt(data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast2 = fit2.forecast(12).rename("Holts Linear Trend")                                                      #previsão 12 meses a frente
# plt.figure()
# fcast2.plot(marker=".", color="blue", legend=True)
# fit2.fittedvalues.plot(marker=".", color="red")

#AJUSTES ARIMA
# plt.figure()
# plot_acf(data)
# plt.figure()
# plot_pacf(data)

model = ARIMA(data, order=(8,2,10))
model_fit = model.fit()
print(model_fit.summary())

# plt.figure()
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.figure()
# residuals.plot(kind="kde")
# print(residuals.describe())

auto_model = auto_arima(data, start_p=1, start_q=1, max_p=6, max_q=6, m=12, 
                        start_P=0, seasonal=True, d=1, D=1, trace=True,
                        error_action="ignore", suppress_warnings=True, 
                        stepwise=True)
