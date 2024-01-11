import pandas as pd
import numpy as np
import os
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf


os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

#Parametros
escola = 35132287           #Codigo da escola a ser analisada

#Data Loading
data = pd.read_csv("enem_2019_tratado.csv", encoding="latin1", sep=",", encoding_errors="replace")

#Data description
print("Brief description of data:")
print(data.head(10))
print(data.loc[:,["IDADE","NU_INSCRICAO","NO_MUNICIPIO_ESC"]].describe())
print("\nColumns names are {}.".format(data.columns))
print("\nData shape is {}.".format(data.shape))
print("\nData info types:\n{}".format(data.dtypes))
print(data.isnull().sum())

#Data by school
schools = data.groupby("CO_ESCOLA").count().loc[:,["NU_INSCRICAO"]]
schools = schools.sort_values(by=["NU_INSCRICAO"], ascending=False)
print("------------------")
print("\nBrief description of data:")
print(schools.head(10))
print("\nSchools shape is {}.".format(schools.shape))

#Select a specific school
data_spec = data.loc[data["CO_ESCOLA"] == escola]
print("------------------")
print(data_spec.loc[:,["IDADE","NU_INSCRICAO","NO_MUNICIPIO_ESC"]].describe())
print("\nData shape is {}.".format(data_spec.shape))
print(data_spec.isnull().sum())

#Regression
variaveis = data_spec[["NOTA_REDACAO","COMP2","COMP4","COMP5"]]
print("\nCorrelação entre variáveis selecionadas:")
print(variaveis.corr())

modelo = smf.ols("NOTA_REDACAO ~ COMP2 + COMP4 + COMP5", data=data_spec)
results = modelo.fit()
residuos = results.resid

