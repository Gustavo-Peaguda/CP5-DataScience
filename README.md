# Checkpoint 5 — Expectativa de vida ao nascer (Random Forest, XGBoost e LightGBM)

**Disciplina:** Data Science & Statistical Computing — FIAP 2026
**Professor:** Jones Egydio
**Entrega:** 01/10/2026

## Integrantes

| Nome | RM |
|---|---|
| Enzo Fernandes Ramos | RM 563705 |
| Felipe Henrique de Souza Cerazi | RM 562746 |
| Gustavo Peaguda de Castro | RM 562923 |
| Lorenzo Andolfatto Coque | RM 563385 |


---

## Apresentação do projeto

Este projeto estima a **expectativa de vida ao nascer** (em anos) de um país em determinado ano, a partir de características socioeconômicas, demográficas e de gasto em saúde. É um problema de **regressão supervisionada**, em que cada observação é um **país-ano** (2000–2015).

Os algoritmos **Random Forest**, **XGBoost** e **LightGBM** são comparados sob o **mesmo protocolo experimental**: mesmo conjunto de teste, mesmos folds de validação cruzada, mesma métrica principal (**RMSE**, em anos) e mesma preparação dos dados. Cada algoritmo é avaliado em três configurações (baseline, Grid Search e Optuna), totalizando nove configurações.

**O que é tratado no projeto (seções do notebook, seguindo o enunciado):**

| Seção | Conteúdo |
|---|---|
| 4.1 | Definição do problema, alvo, fonte dos dados e métrica |
| 4.2 | Diagnóstico, data wrangling e análise exploratória |
| 4.3 | Escolha de variáveis e desenho experimental (separação treino/teste e validação cruzada) |
| 4.4 | Baselines dos três modelos |
| 4.5 | Tuning com Grid Search e Optuna |
| 4.6 | Escolha do melhor modelo e análise de overfitting/underfitting |
| 4.7 | Teste final, consistência, Streamlit e teste de paridade |

### Principais decisões metodológicas

- **Variáveis:** o modelo usa 12 variáveis (10 numéricas e 2 categóricas). Foram retiradas as variáveis de **doenças** (HIV/AIDS, hepatite B, pólio, difteria e sarampo), além de `mortalidade_adulta` (relação direta com o alvo), `obitos_menores_5` (sobreposição com `obitos_infantis`), `imc` (valores implausíveis), `magreza_5_9` (redundante) e `pais` (identificador).
- **Sem vazamento de dados:** a separação treino/teste e a validação cruzada são feitas **por país**, de modo que o teste contém países que o modelo nunca viu. Imputação e codificação ficam dentro de um `Pipeline`, ajustadas apenas com o treino.
- **Teste isolado:** o conjunto de teste só é usado na etapa final, uma única vez. A escolha do modelo usa apenas treino e validação cruzada, por uma regra definida antes (1 erro-padrão; depois menor gap treino-validação; depois menor tempo).
- **Tuning:** Grid Search com 24 combinações por modelo e Optuna com 20 trials por modelo, ambos com os mesmos folds e a mesma métrica.
- **Streamlit:** o aplicativo carrega o **mesmo `Pipeline`** avaliado no notebook (`modelo/modelo_final.joblib`) e inclui uma verificação de paridade notebook × aplicação.

### Resultado de referência

| Item | Valor |
|---|---|
| Configuração escolhida | XGBoost — Optuna |
| RMSE na validação cruzada | 4,09 ± 0,21 anos |
| RMSE no teste | 3,93 anos |
| MAE no teste | 2,95 anos |
| R² no teste | 0,83 |

Os valores finais exatos ficam registrados no notebook e em `modelo/meta.json` após a execução.

---



## Fonte dos dados

Dataset **Life Expectancy (WHO)**, publicado no Kaggle por `kumarajarshi`:
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

Compilado de dados públicos da OMS (Global Health Observatory) e da ONU. Os arquivos já estão na pasta `dados/`.

---

## Como instalar e executar

### 1. Instalar as dependências

Opção A — instalando os pacotes diretamente:

```bash
py -m pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib xgboost lightgbm optuna
```

Opção B — usando o arquivo de dependências:

```bash
py -m pip install -r requirements.txt
```

### 2. Executar o notebook (gera o modelo e os arquivos da pasta `modelo/`)

Abra `Checkpoint05_Expectativa_Vida_RF_XGB_LGBM.ipynb` na **raiz do projeto** e use **Run All**. Isso gera `modelo/modelo_final.joblib`, `modelo/meta.json` e os demais arquivos usados pelo aplicativo.

### 3. Executar a aplicação Streamlit

```bash
py -m streamlit run app.py
```

> O aplicativo precisa dos arquivos gerados pelo notebook (passo 2). Se a pasta `modelo/` já estiver completa no repositório, o passo 2 não é necessário.

### Teste de paridade notebook × Streamlit

Na aplicação, abra a seção **Verificação técnica: Notebook × Streamlit**. Ela recalcula a previsão de 7 observações do teste de consistência com o `Pipeline` carregado e confirma que coincidem com as previsões do notebook.
