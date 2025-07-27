# Previsão de Casos de Dengue

## Introdução

Este trabalho desenvolve modelos de aprendizado profundo para previsão de casos de dengue no estado do Rio de Janeiro, utilizando séries temporais de dados climáticos e técnicas de deep learning. O estudo combina duas abordagens complementares:  

1. **Classificação binária** para detecção de semanas com risco de transmissão  
2. **Modelagem de regressão** para estimativa quantitativa de casos  

A análise focaliza especialmente na sensibilidade dos modelos a diferentes variáveis meteorológicas, empregando permutação controlada para identificar os fatores ambientais mais determinantes nos padrões de transmissão.

---

## 1. Análise Exploratória

A análise inicial revelou características importantes do conjunto de dados. A distribuição de casos mostrou-se altamente assimétrica, com 78% das semanas sem registros de dengue. As correlações lineares identificaram que variáveis pluviométricas, como a chuva acumulada em 21 dias (RAIN_ACC_21) e a média móvel de 28 dias (RAIN_MM_28), apresentaram as associações positivas mais fortes, embora bem modestas (valores em torno de 0,032).

Já fatores térmicos, como a amplitude diária de temperatura (TEMP_RANGE) e a temperatura máxima (TEM_MAX), mostraram correlações negativas fracas. Esses resultados iniciais sugeriram que as relações entre clima e dengue podem envolver padrões não-lineares ou defasagens temporais.

---

## 2. Modelagem: Classificação Binária

Classificar a presença ou ausência de casos de dengue em uma determinada semana. Antes, executar simplesmente a regressão não estava oferecendo bons resultados.

### Arquitetura e Métricas  

O modelo de classificação binária, baseado em arquitetura LSTM com janela de 7 dias, demonstrou excelente capacidade discriminatória. No entanto, apresentou um viés conservador, com alta precisão (75%) mas recall moderado (36%), indicando uma tendência a priorizar a redução de falsos positivos em detrimento da identificação de todos os eventos reais. Esse perfil é adequado para sistemas de alerta onde falsos alarmes geram custos operacionais significativos, mas não notificar casos de dengue quando realmente há um caso não é o ideal para a saúde pública.

---

## 3. Modelagem: Regressão

Prever a magnitude do número de casos de dengue.

### Modelos testados

| Modelo   | Arquitetura                     |
|----------|----------------------------------|
| LSTM     | LSTM + Dropout + Dense          |
| Conv1D   | Conv1D + Pooling + Flatten + Dense |

### Estratégia

- Uso de apenas amostras com casos reais (`y > 0`).
- Aplicação de transformação `log(y+1)` para estabilizar variância e melhorar desempenho.
- Avaliação com múltiplas execuções (`n=5`) e diferentes seeds.

### Graficos de dispersão

A maioria dos pontos no gráfico de dispersão encontra-se abaixo da linha ideal, o que indica que o modelo tende a subestimar os casos de dengue, especialmente em situações em que os valores reais são mais elevados. Isso pode ser visto tanto no gráfico do LSTM quanto no do Conv1D. Nas regiões de baixa magnitude, próximas a 1 caso, observa-se uma alta concentração de pontos, evidenciando a dificuldade do modelo em capturar variações sutis nessa faixa.

Apesar dessas limitações, o padrão geral do gráfico sugere que o modelo é capaz de captar a tendência dos dados, ou seja, mesmo que não acerte com precisão os valores exatos, ele consegue estimar corretamente a ordem de grandeza dos casos. Além disso, a utilização da escala logarítmica foi essencial nesse contexto, pois permite visualizar erros em diferentes ordens de magnitude, o que não seria possível de forma tão clara em uma escala linear.

![alt text](image-3.png) ![alt text](image-4.png)

### Resultados Consolidados de Desempenho

| Métrica       | LSTM (média ± std)      | Conv1D (média ± std)    | Variação Relativa |
|---------------|-------------------------|-------------------------|------------------|
| **MAE**       | 1.7765 ± 0.0130         | 1.9704 ± 0.0394         | +10.9%           |
| **RMSE**      | 4.3759 ± 0.0409         | 4.9259 ± 0.1385         | +12.6%           |
| **R²**        | 0.5712 ± 0.0080         | 0.4563 ± 0.0307         | -20.1%           |
| **MaxError**  | 99.51 ± 1.27            | 119.81 ± 3.31           | +20.4%           |
| **MedianAE**  | 0.5900 ± 0.0149         | 0.6179 ± 0.0212         | +4.7%            |

**Legenda:**

- Valores representam média ± desvio padrão de 5 execuções independentes
- Variação relativa calculada como: (Conv1D - LSTM)/LSTM × 100%

#### Detalhamento por Arquitetura

##### LSTM

- MAE variou entre 1.7629 (min) e 1.8010 (max)
- R² consistente (0.5599 a 0.5804)
- Erro máximo estável (97.82 a 101.48)

##### Conv1D

- MAE mostrou maior dispersão (1.8989 a 2.0171)
- R² mais variável (0.4043 a 0.4993)
- MaxError significativamente maior (115.38 a 124.05)

> **Conclusão**: O modelo **LSTM** superou o **Conv1D** em todas as métricas, mostrando menor erro e maior capacidade de explicação (R²).

---

## 4. Sensibilidade às Variáveis Climáticas

### Técnica: Permutation Importance (ΔMAE)

Cada variável foi embaralhada individualmente e a degradação no MAE foi medida. Quanto maior a ΔMAE, maior a importância da variável.

### Variáveis mais impactantes

**Top 5 variáveis por arquitetura**:  

| Rank | LSTM                | Conv1D              |
|------|---------------------|---------------------|
| 1    | EXTREME_RAIN (0.0229)| RAIN_MM_28 (0.0825) |
| 2    | SIGNIFICANT_RAIN (0.0084)| TEMP_RANGE (0.0561)|
| 3    | RAIN_MM_21 (0.0064) | TEMP_RANGE_MM_28 (0.0276)|
| 4    | TEMP_RANGE (0.0061) | RAIN_ACC_21 (0.0272)|
| 5    | TEMP_RANGE_MM_14 (0.0060)| TEM_AVG_MM_21 (0.0249)|

A análise de importância por permutação trouxe visões valiosas sobre como os modelos utilizam as variáveis climáticas. No LSTM, a chuva extrema (EXTREME_RAIN) emergiu como o fator mais impactante, enquanto no Conv1D a média móvel de chuva em 28 dias (RAIN_MM_28) mostrou-se mais relevante. A amplitude térmica diária (TEMP_RANGE) manteve importância significativa em ambos os modelos, corroborando os achados da análise exploratória. É interessante notar que algumas variáveis com correlações lineares fracas, como EXTREME_RAIN, revelaram-se críticas na permutação, indicando que os modelos capturaram relações não-lineares não evidentes na análise inicial.

---

## 5. Conclusão Geral

Os resultados demonstram que o modelo LSTM é a escolha mais adequada para o sistema de previsão, oferecendo maior precisão e estabilidade. A análise de sensibilidade climática sugere que diferentes aspectos das condições meteorológicas afetam a transmissão da dengue: eventos extremos no curto prazo (LSTM) e acumulados no médio prazo (Conv1D).

### Direções Futuras  

- Incorporar defasagens biológicas customizadas (2-8 semanas)
- Fazer engenharia de features
- Testar modelos híbridos
- Integrar variáveis socioambientais (índice de urbanização, histórico de campanhas)  
