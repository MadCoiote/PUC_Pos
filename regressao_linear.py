# -*- coding: utf-8 -*-
"""
Exercicio da matéria de pós-graduação PUC Minas
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

data = pd.read_csv("estudo_caso_02.csv")
print(data.head())
#data.hist()

#Info from Years of Experience
print("\nMédia de experiência em anos: {:.2f}.".format(data["YearsExperience"].mean()))
print("Desvio Padrão de experiência em anos: {:.2f}.".format(data["YearsExperience"].std()))

#Info from Years of Experience
print("\nMédia de salário anual em dólares: {:.0f}.".format(data["Salary"].mean()))
print("Desvio padrão de salário anual em dólares: {:.0f}.".format(data["Salary"].std()))

#General Info
print(data.describe())
print("\nCovariância dos Dados")
print(data.cov())

print("\nCorrelação dos Dados")
print(data.corr())

# plt.scatter(data["YearsExperience"], data["Salary"])
# plt.xlabel("Anos de Experiência")
# plt.ylabel("Salário Anual em Dólares")

Y = data["Salary"]
X1 = data["YearsExperience"]
X2 = sm.add_constant(data["YearsExperience"])
model1 = sm.OLS(Y,X1)
results1 = model1.fit()
model2 = sm.OLS(Y,X2)
results2 = model2.fit()
print("\nParâmetros sem Interseção\n{}".format(results1.params))
print("\nParâmetros com Interseção\n{}".format(results2.params))
print(results1.summary())
print(results2.summary())