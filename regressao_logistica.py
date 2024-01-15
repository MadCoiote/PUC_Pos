import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

#ETL
doenca_pre = pd.read_csv("casos_obitos_doencas_preexistentes.csv", sep=";", encoding="utf-8")
doenca_pre = doenca_pre.dropna(subset=["cs_sexo"])                                                      #eliminar valores NaN
doenca_pre["cs_sexo"] = doenca_pre["cs_sexo"].astype("category")
doenca_pre["obito"] = doenca_pre["obito"].astype("category")

relacao = doenca_pre.loc[(doenca_pre["cs_sexo"]!="IGNORADO") & (doenca_pre["cs_sexo"]!="INDEFINIDO")]   #eliminar categorias diferentes de masculino e feminino

#DESCRICAO DOS DADOS
print("DADOS DE COVID DO ESTADO DE SÃO PAULO APOS ETL")
print("Início do data frame:")
print(relacao.head(10))
print("\nNomes dos campos e tipo de Dados:")
print(relacao.dtypes)
print("\nEstatisticas dos dados numericos:")
print(relacao.describe())
print("\nValores NaN nas colunas:")
print(relacao.isnull().sum())
print("---------------\n")

print("Valores na coluna relacionada a sexo das pessoas:")
print(relacao["cs_sexo"].value_counts())

#REGRESSAO
modelo1 = smf.glm(formula="obito~cs_sexo", data=relacao, family=sm.families.Binomial())
results1 = modelo1.fit()
print(results1.summary())

razao_de_chance = np.exp(results1.params[3])
