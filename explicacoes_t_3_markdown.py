# Células Markdown para inserir no T3.ipynb

# (Logo após a importação das bibliotecas)
"""
## Introdução

Este notebook apresenta um modelo preditivo baseado em deep learning para a previsão semanal de casos de dengue no estado do Rio de Janeiro entre os anos de 2018 e 2023. Os modelos foram desenvolvidos com base em séries temporais de variáveis climáticas e histórico de casos, utilizando redes neurais recorrentes (LSTM) e convolucionais (Conv1D).

A abordagem é dividida em três etapas principais:
1. Exploração e preparação dos dados;
2. Treinamento de modelos para classificação binária (existência ou não de casos);
3. Treinamento de modelos para regressão (quantidade de casos).

Este notebook documenta o pipeline completo, desde a análise exploratória até a investigação da importância das variáveis.
"""

# (Antes da seção de análise exploratória)
"""
## Análise Exploratória

Nesta etapa, foi realizada uma inspeção das variáveis presentes no dataset, bem como a análise da distribuição dos casos de dengue nos conjuntos de treino, validação e teste.

Também foram calculadas as correlações entre variáveis climáticas e o número de casos, com o objetivo de identificar quais variáveis apresentavam relação linear mais forte com os casos de dengue.
"""

# (Antes do treinamento do modelo de classificação binária)
"""
## Modelo de Classificação Binária

Nesta etapa, o objetivo é prever se haverá ou não ao menos um caso de dengue em uma dada semana. Para isso, foi utilizada uma rede LSTM treinada com uma janela deslizante de 14 dias.

O modelo foi avaliado utilizando métricas como acurácia, precision, recall, F1 score, AUC-ROC e AUC-PR, bem como por meio da matriz de confusão. Observou-se que o modelo possui um comportamento conservador, preferindo minimizar falsos positivos. Isso se reflete em um alto precision e baixo recall.

Curvas ROC e Precision-Recall também foram geradas para avaliar a calibração do modelo.
"""

# (Antes da seção de regressão)
"""
## Modelo de Regressão

O objetivo desta etapa é prever a quantidade de casos de dengue. Utilizamos apenas semanas com pelo menos 1 caso registrado para treinar os modelos.

Foram aplicadas duas arquiteturas: LSTM e Conv1D, ambas utilizando uma janela de 7 dias de variáveis climáticas e epidemiológicas. O treino foi realizado em cinco execuções distintas com seeds diferentes, permitindo a avaliação da robustez dos modelos.

As métricas utilizadas incluem MAE, RMSE, R², erro máximo e erro absoluto mediano. Os resultados apontam que o modelo LSTM apresentou desempenho superior ao Conv1D em todas as métricas avaliadas.
"""

# (Antes da visualização de dispersão dos resultados)
"""
## Gráfico de Dispersão

A dispersão dos resultados do modelo LSTM pode ser visualizada no gráfico abaixo. O eixo x representa os valores reais de casos, enquanto o eixo y representa os valores previstos. Ambos os eixos estão em escala logarítmica, para melhor visualizar a dispersão em faixas amplas de valores.

A linha vermelha representa a previsão ideal (valores previstos = valores reais). O afastamento dos pontos em relação a essa linha indica o erro de previsão. Observa-se que a maioria das previsões tende a subestimar os valores reais, o que é coerente com a dificuldade do modelo em capturar picos epidêmicos.
"""

# (Antes da parte de importância de variáveis)
"""
## Análise de Importância das Variáveis

Para investigar a sensibilidade do modelo às variáveis climáticas, foi realizada uma análise de importância por permutação. Isso consiste em embaralhar os valores de uma variável de entrada e observar o impacto no erro de previsão (MAE).

A métrica utilizada é o aumento do MAE em relação ao baseline (sem embaralhamento). Valores maiores indicam maior importância da variável na capacidade preditiva do modelo. Esta abordagem foi aplicada tanto para o modelo LSTM quanto para o Conv1D.
"""
