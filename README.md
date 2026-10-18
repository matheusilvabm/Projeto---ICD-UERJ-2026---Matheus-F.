# Predição de falhas de máquinas em 24 horas (manutenção preditiva)

Projeto de **aprendizado supervisionado (classificação binária)** para prever, a cada hora, se uma máquina industrial vai falhar (em qualquer componente) nas **próximas 24 horas**, usando telemetria, erros e histórico de manutenção.

**Autor(a):** _Matheus Fonseca Da Silva_ · **Disciplina:** _ICD (UERJ - 2026)_

> **Resumo dos resultados:** os modelos de árvores (Random Forest e Gradient Boosting) atingem PR-AUC ≈ 0,999 e F1 ≈ 0,99 no teste final, contra PR-AUC 0,018 do baseline aleatório e 0,48 de uma regra simples de erros. **Esse desempenho é artificialmente alto**: os dados apresentam um padrão típico de simulação (veja [Limitações](#limitações)), e por isso os números medem a qualidade do pipeline, não o desempenho esperado em campo.

---

## 1. Problema e dataset

- **Pergunta:** dado o histórico recente de uma máquina na hora *t*, ela falha em `(t, t + 24h]`?
- **Dataset:** [Microsoft Azure Predictive Maintenance (Kaggle)](https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance): 100 máquinas observadas durante 2015, com telemetria horária (tensão, rotação, pressão, vibração), erros, manutenções, falhas e características das máquinas (modelo e idade).
- **Tamanho:** 876.100 leituras horárias, 761 falhas, sem valores nulos nem duplicatas.
- **Desbalanceamento:** apenas ~2% das horas antecedem uma falha (1,96% das linhas após a construção do alvo). Por isso a acurácia não é usada; a métrica principal é a **PR-AUC**, além de precisão, recall e F1.

## 2. Pipeline

| Etapa | O que foi feito |
|---|---|
| Auditoria e consistência | Verificação de nulos, duplicatas, IDs inexistentes, datas fora do período, valores impossíveis e regularidade horária. Outliers (≈1% por sensor) foram **mantidos**, pois podem ser sinal de falha. |
| Alvo | `target = 1` se há falha em qualquer componente estritamente depois de *t* e até *t + 24h*. |
| Features (62) | Somente passado: estatísticas móveis dos sensores (3h, 12h, 24h), contagem de erros (24h, 72h), horas desde a última manutenção de cada componente e da última falha, modelo (one-hot) e idade. |
| Escalonamento | `StandardScaler` apenas na Regressão Logística, dentro do pipeline (aprendido só no treino de cada fold). |
| Validação | **Validação cruzada temporal** com janela expansiva (3 folds: jul, ago, set), com embargo de 24h, e **teste final** de outubro a dezembro, usado uma única vez. Nada é escolhido com base no teste. |
| Baselines | `DummyClassifier` e uma regra simples ("pelo menos 1 erro nas últimas 24h"). |
| Modelos | Regressão Logística, Random Forest e Gradient Boosting (`HistGradientBoosting`), com pesos de classe para o desbalanceamento. |
| Otimização | `RandomizedSearchCV` com a mesma validação temporal, pontuado por PR-AUC. |
| Limiar | Escolhido em previsões fora da amostra (jul-set), por máximo F1 e por mínimo custo. |
| Análise crítica | Métrica por evento, experimentos de remoção de features (ablação), importância por permutação e análise dos erros. |

## 3. Resultados

### Comparação dos modelos

| Modelo | PR-AUC CV (média ± desvio) | PR-AUC teste | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| Baseline: Dummy | 0,0188 ± 0,0002 | 0,0183 | 0,018 | 0,019 | 0,018 |
| Baseline: regra de erros | 0,4638 ± 0,0387 | 0,4812 | 0,195 | 1,000 | 0,326 |
| Regressão Logística | 0,8384 ± 0,0329 | 0,8319 | 0,760 | 0,941 | 0,841 |
| Random Forest | 0,9992 ± 0,0009 | 0,9992 | 0,987 | 0,997 | 0,992 |
| **Gradient Boosting** (escolhido pela CV) | **0,9998 ± 0,0002** | **0,9994** | **0,980** | **1,000** | **0,990** |

Precisão, recall e F1 no limiar de máximo F1 escolhido na validação. A prevalência no teste é 1,83%.

![Comparação dos modelos](imagens/comparacao.png)

- **Random Forest e Gradient Boosting estão praticamente empatados**; a diferença está dentro da variação entre folds.
- **As árvores superam o modelo linear** (0,999 contra 0,84), o que indica relação não linear entre as variáveis e a falha.
- **Os hiperparâmetros influenciam pouco**: as combinações testadas ficam muito próximas.

### Gradient Boosting no teste final (limiar de máximo F1)

- Matriz de confusão: 4.050 acertos, **0 falhas não avisadas** e 84 alarmes falsos (de ~217 mil linhas normais).
- **Por evento:** 171 de 171 falhas avisadas, com antecedência mediana de 24h (que é o teto definido pelo alvo) e cerca de 0,009 linhas de alarme falso por máquina-dia.

![Matrizes de confusão](imagens/confusao.png)

### De onde vem o desempenho (ablação, PR-AUC no teste)

| Cenário | PR-AUC |
|---|---|
| Todas as features | 0,999 |
| Sem manutenção | 0,999 |
| Sem sensores | 0,976 |
| Sem erros | 0,896 |
| Sem erros e sem manutenção | 0,784 |
| Só erros | 0,706 |
| Só sensores | 0,223 |

![Ablação](imagens/ablacao.png)

O grupo **erros** é o que mais sustenta o desempenho: a feature `err_total_sum_24h` domina a importância por permutação (queda de ≈ 0,53 na PR-AUC), e os sensores sozinhos ficam muito abaixo. A manutenção não acrescenta nada quando os demais grupos estão presentes.

![Importância das features](imagens/importancia.png)

## Limitações

- **Os dados parecem simulados.** O estudo de evento mostra que, 48 horas antes de toda falha, os quatro sensores mudam de uma vez e permanecem num patamar constante, comportamento que máquinas reais não costumam apresentar. Além disso, a regra "pelo menos 1 erro nas últimas 24h" detecta 100% das linhas pré-falha. O problema é, portanto, **artificialmente fácil**, e resultados próximos de 100% **não devem ser extrapolados** para operação real.

  ![Comportamento pré-falha](imagens/evento.png)

- **Falsos positivos:** apenas 11 dos 84 (13%) ocorrem entre 24h e 72h antes de uma falha real; os demais não têm falha próxima. Há episódios de alerta com probabilidade ≈ 1,0 sem falha por perto (por exemplo, máquina 20 em 14/12/2015 e máquina 92 em 18/10/2015) que merecem investigação.
- **Alvo único para qualquer componente.** Um modelo por componente seria mais acionável.
- **Teste com 171 falhas**, sem intervalos de confiança calculados.
- **Sensibilidade do limiar:** o limiar de mínimo custo escolhido na validação teve custo maior que o de máximo F1 no teste (103 contra 84).

## Próximos passos

- Validar em dados reais (por exemplo, MetroPT-3) para medir o desempenho sem precursores artificiais.
- Testar outros horizontes (12h, 48h, 72h) e modelagem por componente.
- Calibrar probabilidades e ajustar os custos de erro à operação real.
- Calcular intervalos de confiança (*bootstrap* por máquina).

---

## Como executar

### Estrutura do repositório

```
.
├── README.md
├── predicao_falhas_24h.ipynb     # notebook completo, já executado (com saídas)
├── requirements.txt
├── imagens/                      # figuras usadas neste README
└── data/                         # coloque aqui os 5 CSVs (não versionados)
    ├── PdM_telemetry.csv
    ├── PdM_errors.csv
    ├── PdM_maint.csv
    ├── PdM_failures.csv
    └── PdM_machines.csv
```

### Passo a passo (VS Code)

1. Instale o Python e, no VS Code, as extensões **Python** e **Jupyter**.
2. Baixe os 5 CSVs no [Kaggle](https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance) e coloque-os em `data/` com os nomes acima.
3. Abra a pasta do projeto no VS Code e, no terminal, crie o ambiente e instale as dependências:

   Windows:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Mac/Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Abra `predicao_falhas_24h.ipynb`, selecione o kernel do `.venv` e clique em **Executar Tudo**.

### Observações

- **Reprodutibilidade:** as sementes aleatórias são fixas (`RANDOM_STATE = 42`), então os resultados devem ser praticamente idênticos entre execuções.
- **Tempo:** a busca de hiperparâmetros levou cerca de 12 minutos no computador usado (Random Forest ≈ 7 min, Gradient Boosting ≈ 5 min, Regressão Logística ≈ 1 min), e o notebook completo leva mais que isso. Se estiver lento, defina `MODO_RAPIDO = True` na célula de configuração.
- **Erros comuns:** `FileNotFoundError` indica CSV ausente ou com nome diferente em `data/`; `ModuleNotFoundError` indica que o kernel selecionado não é o do `.venv`.

## Tecnologias

Python, pandas, NumPy, scikit-learn, matplotlib e seaborn.
