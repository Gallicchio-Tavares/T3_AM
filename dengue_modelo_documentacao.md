# Previsão de Casos de Dengue com Deep Learning (LSTM e Conv1D)

## Dados utilizados

- **Fonte**: Dados epidemiológicos (SINAN) + meteorológicos (ERA5).
- **Período**: 2018 a 2023.
- **Escala**: Dados semanais para unidades de saúde do Estado do Rio de Janeiro.
- **Formato**: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` em `.pickle`.
- **Estrutura**: `X`: (n amostras, n features), `y`: (n amostras,)

## Objetivo do projeto

Desenvolver um pipeline de previsão de dengue com foco em:

1. Prever a **presença/ausência** de casos (classificação binária);
2. Prever a **quantidade de casos** (regressão);
3. Avaliar a **sensibilidade às variáveis climáticas**.

## Análise exploratória

- O dataset é altamente **desbalanceado**: 96,7% dos registros possuem 0 casos.
- Correlações com os casos mostram que variáveis como:
  - `TEM_AVG`, `RAIN`, `RH_AVG`, `IDEAL_TEMP`, `EXTREME_TEMP`, `SIGNIFICANT_RAIN`
  - têm relação com os aumentos nos casos de dengue.

## Pipeline 1 – Classificação binária

### Transformação da variável-alvo

```python
y_bin = (y > 0).astype(int)
```

### Pré-processamento

- Criação de janelas temporais (`window_size = 7`)
- Normalização com `MinMaxScaler`.

### Modelo LSTM (classificação)

- LSTM + Dropout + Dense (sigmoid)
- Loss: `binary_crossentropy`
- Métricas: Accuracy, Precision, Recall, F1, AUC

### Avaliação (exemplo)

- Accuracy: 0.93
- F1: 0.50
- AUC-ROC: 0.90
- AUC-PR: 0.61
- Problemas: baixa sensibilidade para casos positivos, mas boa para negativos.

## Pipeline 2 – Regressão (previsão da magnitude)

### Foco

Prever a quantidade de casos (valor contínuo), **apenas onde há casos (>0)**.

### Pré-processamento

- Seleção apenas de variáveis **climáticas e ambientais** (sem lags de casos).
- Normalização com `MinMaxScaler`.
- Criação de janelas (`window_size = 7`)
- **Filtragem das janelas com `y > 0`**

### Modelo Conv1D e LSTM (regressão)

- Loss: `Huber`
- Métricas: MAE, RMSE, R²

### Avaliação

- MAE: ~1.73
- RMSE: ~4.26
- R²: ~0.59
- Gráficos: dispersão em escala log, resíduos vs predições

### Diagnóstico

- O modelo tem **dificuldade em prever picos elevados**.
- Erros maiores que o esperado indicam presença de **outliers e alta variabilidade**.
- Excluir variáveis relacionadas a casos anteriores e focar em clima ajudou a reduzir overfitting.

## Próximos passos

