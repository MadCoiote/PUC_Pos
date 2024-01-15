import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats.distributions import norm

os.chdir('/home/hernandes/Documentos/Cursos/12 - Pos Graduação PUC Minas/13 - Modelos Estatísticos')

vendas = pd.read_excel("comissao.xlsx")
print(vendas.head(10))
print("\n",vendas.describe())

# plt.scatter(vendas["quantidade"],vendas["comissao"],color="black",marker="o")
# plt.xlabel("Quantidade de itens vendidos")
# plt.ylabel("Comissão em reais")
# plt.grid(True)


# O Teste de Shapiro-Wilk tem como objetivo avaliar se uma distribuição é semelhante a uma distribuição normal. 

# plt.figure()
#stats.probplot(vendas["comissao"], dist="norm", plot=plt)

estat, p = stats.shapiro(vendas["comissao"])
print("Resultado do testes de Shapiro-Wilk mostra que ao rejeitar a hipótese de distribuição normal, a chance de erro é {:.2%}".format(p))

#Só para mostrar que o teste de Shapiro não faz sentido nesse contexto
X = np.arange(0,50)
Y = norm(25,5).pdf(X)
estat, p = stats.shapiro(Y)
print("--------")
print("Resultado do testes de Shapiro-Wilk mostra que ao rejeitar a hipótese de distribuição normal, a chance de erro é {:.2%}".format(p))
######################

plt.show()