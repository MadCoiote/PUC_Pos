import pandas as pd
import os
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import scipy.stats as stats

os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

#Parametros
escola = 35104985           #Codigo da escola a ser analisada  original é 35132287

#Data Loading
data = pd.read_csv("enem_2019_tratado.csv", encoding="latin1", sep=",", encoding_errors="replace")

#Data description
print("Brief description of general data:")
print(data.head(10))
print(data.loc[:,["IDADE","NU_INSCRICAO","NO_MUNICIPIO_ESC"]].describe())
print("\nColumns names are {}.".format(data.columns))
print("\nData shape is {}.".format(data.shape))
print("\nData info types:\n{}".format(data.dtypes))
print("\nNan values in each field:")
print(data.isnull().sum())

#Data by school
schools = data.groupby("CO_ESCOLA").count().loc[:,["NU_INSCRICAO"]]
schools = schools.sort_values(by=["NU_INSCRICAO"], ascending=False)
print("------------------")
print("\nBrief description of students in each school:")
print(schools.head(10))
print("\nSchools shape is {}.".format(schools.shape))

#Select a specific school
data_spec = data.loc[data["CO_ESCOLA"] == escola]
print("------------------")
print("\n\nFrom here on all the data is restricted to school {}.\n\n".format(escola))
print(data_spec.loc[:,["IDADE","NU_INSCRICAO","NO_MUNICIPIO_ESC"]].describe())
print("\nData shape is {}.".format(data_spec.shape))
print("\nNan values in each field:")
print(data_spec.isnull().sum())

#Regression
variaveis = data_spec[["NOTA_REDACAO","COMP2","COMP4","COMP5"]]
print("\nCorrelação entre variáveis selecionadas:")
print(variaveis.corr())

modelo = smf.ols("NOTA_REDACAO ~ COMP2 + COMP4 + COMP5", data=data_spec)
results = modelo.fit()
residuos = results.resid

#Teste de normalidade dos resíduos (Shapiro): se p<0.05 a hipótese de distribuição normal é rejeitada
estat, p = stats.shapiro(residuos)
print("\nProbabilidade de erro ao rejeitar a distribuição normal dos resíduos é de p = {:.2%}.".format(p))
stats.probplot(residuos, dist="norm",plot=plt)

#Teste de homocedasticidade dos resíduos (Breush-Pagan): se p<0.05 a hipótese de homocedasticidade é rejeitada
estat, p, f, fp = sms.het_breuschpagan(residuos, results.model.exog)
print("Probabilidade de erro ao rejeitar a homocedasticidade dos resíduos é de p = {:.2%}.".format(p))
plt.figure()
plt.scatter(y=residuos, x=results.predict(), color="black", alpha=0.5)

#Verificar a existência de outliers nos resíduos
outliers = results.outlier_test()
print("Na análise de outliers, o mínimo foi de {:.2f} e o máximo de {:.2f}.".format(outliers["student_resid"].min(),outliers["student_resid"].max()))

#Verificar multicolinearidade: Fator de Inflação de Variância (VIF): se VIF menor que 5 pode-se considerar a ausência de colinearidade
vif = add_constant(data_spec[["COMP2","COMP4","COMP5"]])
vif_comp = pd.Series([variance_inflation_factor(vif.values, j) for j in range(vif.shape[1])], index=vif.columns)
print("\nResultado do VIF:")
print(vif_comp)

#Regressão
print(results.summary())

#plt.show()