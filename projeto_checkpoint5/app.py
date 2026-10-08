"""Aplicação Streamlit — Checkpoint 5: expectativa de vida ao nascer (Random Forest, XGBoost, LightGBM).

O app apenas LÊ o que o notebook gerou (modelo/ e dados/): não treina, não faz tuning, não refaz validação
cruzada e não recria nenhum pré-processamento. A previsão usa o Pipeline final salvo
(imputação + codificação + modelo) via joblib.load(...).predict(...).

Paridade com o notebook: os casos do teste de consistência são reproduzidos pelo Pipeline carregado e comparados
com a previsão salva pelo notebook, com tolerância de ponto flutuante (np.allclose; valores em meta.json).
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

st.set_page_config(page_title="Expectativa de vida — Checkpoint 5", layout="wide")

# ----------------------------------------------------------------------------
# Arquivos gerados pelo notebook (Seção 17)
# ----------------------------------------------------------------------------
PASTA_MODELO, PASTA_DADOS = Path("modelo"), Path("dados")
CAM_MODELO = PASTA_MODELO / "modelo_final.joblib"
CAM_META = PASTA_MODELO / "meta.json"
CAM_BASE = PASTA_DADOS / "base_tratada.csv"
CAM_COMPARACAO = PASTA_MODELO / "comparacao_configuracoes.csv"
CAM_IMP_PERM = PASTA_MODELO / "importancia_permutacao.csv"
CAM_PRED_TESTE = PASTA_MODELO / "predicoes_teste.csv"

obrigatorios = [CAM_MODELO, CAM_META, CAM_BASE, CAM_COMPARACAO, CAM_IMP_PERM, CAM_PRED_TESTE]
faltando = [str(c) for c in obrigatorios if not c.exists()]
if faltando:
    st.error("Arquivos não encontrados: " + ", ".join(f"`{c}`" for c in faltando) +
             ". Execute o notebook do início ao fim (Run All) para gerá-los e abra o app a partir da pasta do projeto.")
    st.stop()


@st.cache_resource
def carregar_modelo():
    # Pipeline final (pré-processamento + modelo), exatamente como avaliado no teste do notebook
    return joblib.load(CAM_MODELO)


@st.cache_data
def carregar_meta():
    with open(CAM_META, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_csv(caminho):
    return pd.read_csv(caminho)


@st.cache_data
def para_csv(d):
    return d.to_csv(index=False).encode("utf-8")


modelo_final = carregar_modelo()
meta = carregar_meta()
base = carregar_csv(CAM_BASE)
comparacao = carregar_csv(CAM_COMPARACAO)
imp_perm = carregar_csv(CAM_IMP_PERM)
pred_teste = carregar_csv(CAM_PRED_TESTE)

# A exploração de dados usa SOMENTE o conjunto de treino (como a EDA do notebook): os países do teste ficam fora.
if "conjunto" in base.columns:
    base_treino = base[base["conjunto"] == "treino"].copy()
else:  # compatibilidade: remove os países do teste, identificados no arquivo de predições do notebook
    base_treino = base[~base["pais"].isin(set(pred_teste["pais"]))].copy()

FEATURES = meta["features"]                       # mesmas colunas, na mesma ordem, usadas no treino do Pipeline
NUM, CAT = meta["features_numericas"], meta["features_categoricas"]
ALVO, UNID = meta["alvo"], meta["unidade_alvo"]
ROT = meta["rotulos"]
info_var = {r["variavel"]: r for r in meta["dicionario_variaveis"]}
STATUS_PT = {"Developed": "Desenvolvido", "Developing": "Em desenvolvimento"}
m_teste = meta["metricas_teste"]
hp = meta.get("hiperparametros") or {}
MODELO_NOME, ESTRATEGIA = meta["modelo"], meta["estrategia"]


def fmt(x, casas=2):
    """Formata número no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def milhar(n):
    return f"{int(n):,}".replace(",", ".")


# ----------------------------------------------------------------------------
# Barra lateral: projeto, modelo final, métricas e filtros (filtros afetam SÓ a exploração)
# ----------------------------------------------------------------------------
st.sidebar.title("Expectativa de vida ao nascer")
st.sidebar.caption("Checkpoint 5 — Data Science & Statistical Computing — FIAP")

with st.sidebar.expander("Sobre o projeto", expanded=False):
    st.markdown(f"**Pergunta preditiva:** {meta['pergunta']}")
    st.markdown(f"**Fonte dos dados:** {meta['fonte']}")
    st.markdown(f"**Variável-alvo:** `{ALVO}`  \n**Unidade:** {UNID}")
    st.markdown(f"**Observações:** {milhar(meta['n_observacoes'])} país-ano  \n**Países:** {meta['n_paises']}")

with st.sidebar.expander("Modelo final", expanded=True):
    st.markdown(f"**{MODELO_NOME}**  \nConfiguração: **{ESTRATEGIA}**")
    if hp:
        st.caption("Hiperparâmetros: " + ", ".join(f"{k} = {v}" for k, v in hp.items()))
    else:
        st.caption("Hiperparâmetros: padrão definido no Baseline (300 árvores).")
    st.caption("Escolhido pela regra do protocolo (1 erro-padrão, depois menor gap treino-validação), "
               "usando só treino/validação. O teste não participou da escolha.")

st.sidebar.markdown("**Desempenho no teste isolado**")
st.sidebar.markdown(f"RMSE: **{fmt(m_teste['RMSE'])}** anos  \nMAE: **{fmt(m_teste['MAE'])}** anos  \nR²: **{fmt(m_teste['R2'], 3)}**")

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
st.sidebar.caption("Afetam somente a aba “Explorar os dados” (conjunto de treino). Modelo, métricas e diagnóstico não mudam.")
status_opc = sorted(base_treino["status"].dropna().unique())
regiao_opc = sorted(base_treino["regiao"].dropna().unique())
status_sel = st.sidebar.multiselect("Status", status_opc, default=status_opc, format_func=lambda v: STATUS_PT.get(v, v))
regiao_sel = st.sidebar.multiselect("Região", regiao_opc, default=regiao_opc)
df = base_treino[base_treino["status"].isin(status_sel) & base_treino["regiao"].isin(regiao_sel)]
if df.empty:
    st.sidebar.warning("Nenhuma observação com esses filtros — mostrando todo o conjunto de treino.")
    df = base_treino

# ----------------------------------------------------------------------------
# Cabeçalho e abas
# ----------------------------------------------------------------------------
st.title("Previsão da expectativa de vida ao nascer")
aba_visao, aba_dados, aba_diag, aba_prev = st.tabs(
    ["Visão geral", "Explorar os dados", "Diagnóstico do modelo", "Fazer uma previsão"])

# ============================================================================
# ABA 1 — Visão geral
# ============================================================================
with aba_visao:
    st.markdown("O objetivo é prever a **expectativa de vida ao nascer** de um determinado país-ano utilizando indicadores "
                "socioeconômicos, demográficos, de saúde pública e imunização.")
    st.info("`ano` é utilizado como variável explicativa (uma característica observada do país-ano). "
            "O projeto **não** tem como objetivo prever anos futuros.")

    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Observações (país-ano)", milhar(meta["n_observacoes"]))
    with c2.container(border=True):
        st.metric("Países", meta["n_paises"])
    with c3.container(border=True):
        st.metric("Modelo final", MODELO_NOME, help=f"Configuração: {ESTRATEGIA}")
    with c4.container(border=True):
        st.metric("RMSE de teste", f"{fmt(m_teste['RMSE'])} anos")

    st.subheader("Como o projeto funciona?")
    etapas = [("Dados", "Base OMS/ONU (Kaggle) com região por país"),
              ("Tratamento", "Limpeza determinística; imputação e codificação dentro do Pipeline"),
              ("Comparação dos modelos", "9 configurações, validação cruzada por país"),
              ("Seleção", "Regra definida antes do teste, só com treino/validação"),
              ("Avaliação final", "Teste isolado, usado uma única vez"),
              ("Previsão", "O mesmo Pipeline avaliado faz as novas previsões")]
    for col, (i, (titulo, desc)) in zip(st.columns(len(etapas)), enumerate(etapas, start=1)):
        with col.container(border=True):
            st.markdown(f"**{i}. {titulo}**")
            st.caption(desc)

    st.subheader("Modelos avaliados")
    st.dataframe(pd.DataFrame({"Modelo": ["Random Forest", "XGBoost", "LightGBM"],
                               "Baseline": ["✓"] * 3, "Grid Search": ["✓"] * 3, "Optuna": ["✓"] * 3}), hide_index=True)
    st.caption("Todas as configurações usam o mesmo teste, os mesmos folds (separados por país) e a mesma métrica principal (RMSE).")

    st.subheader("Resultado final")
    st.caption(f"Configuração promovida ao teste: **{meta['configuracao_escolhida']}**. "
               f"Valores calculados uma única vez em {milhar(meta['n_teste'])} observações de países que não participaram do treino nem do tuning.")
    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("RMSE", f"{fmt(m_teste['RMSE'])} anos")
        st.caption("Erro médio, em anos, que dá mais peso aos erros grandes. Quanto menor, melhor.")
    with m2.container(border=True):
        st.metric("MAE", f"{fmt(m_teste['MAE'])} anos")
        st.caption("Erro absoluto médio: o quanto, em anos, a previsão costuma se afastar do valor real.")
    with m3.container(border=True):
        st.metric("R²", fmt(m_teste["R2"], 3))
        st.caption("Fração da variação da expectativa de vida explicada pelo modelo. Quanto mais perto de 1, melhor.")

    with st.expander("Variáveis utilizadas e variáveis excluídas"):
        tabela_var = pd.DataFrame(meta["dicionario_variaveis"])
        tabela_var["Utilizada"] = tabela_var["usada_no_modelo"].map({True: "sim", False: "não"})
        tabela_var = tabela_var[["variavel", "descricao", "unidade", "Utilizada", "justificativa"]]
        tabela_var.columns = ["Variável", "Descrição", "Unidade", "Utilizada", "Justificativa"]
        st.dataframe(tabela_var.sort_values("Utilizada", ascending=False), hide_index=True)

# ============================================================================
# ABA 2 — Explorar os dados (usa os filtros da barra lateral)
# ============================================================================
with aba_dados:
    e1, e2, e3 = st.columns(3)
    with e1.container(border=True):
        st.metric("Observações", milhar(len(df)))
    with e2.container(border=True):
        st.metric("Países", df["pais"].nunique())
    with e3.container(border=True):
        st.metric("Período", f"{int(df['ano'].min())} a {int(df['ano'].max())}")
    st.caption("Base tratada, **somente conjunto de treino** (a mesma usada na análise exploratória do notebook), com os filtros da barra lateral. "
               "Os países do conjunto de teste ficam fora desta aba para preservar a separação usada na avaliação final.")

    st.subheader("Gráficos exploratórios")
    cores = {"Developed": "tab:blue", "Developing": "tab:orange"}

    def dispersao(x, titulo, xlabel, log_x=False, dados=None):
        """Dispersão variável explicativa × expectativa de vida, colorida por status (base de treino filtrada)."""
        origem = df if dados is None else dados
        d = origem[[x, ALVO, "status"]].dropna()
        if log_x:
            d = d[d[x] > 0]                      # escala logarítmica exige valores positivos
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for stt, g in d.groupby("status"):
            ax.scatter(g[x], g[ALVO], s=14, alpha=0.5, color=cores.get(stt), label=STATUS_PT.get(stt, stt))
        if log_x:
            ax.set_xscale("log")
        rho = d[[x, ALVO]].corr(method="spearman").iloc[0, 1] if len(d) > 2 else np.nan
        ax.set_title(f"{titulo}\n(ρ de Spearman = {rho:+.2f}, n = {milhar(len(d))})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Expectativa de vida ao nascer (anos)")
        ax.legend(title="Status")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.hist(df[ALVO], bins=35, color="steelblue", edgecolor="white")
        ax.axvline(df[ALVO].mean(), color="firebrick", ls="--", label=f"Média = {fmt(df[ALVO].mean(), 1)} anos")
        ax.axvline(df[ALVO].median(), color="darkgreen", ls=":", label=f"Mediana = {fmt(df[ALVO].median(), 1)} anos")
        ax.set_title("Como a expectativa de vida se distribui")
        ax.set_xlabel("Expectativa de vida ao nascer (anos)")
        ax.set_ylabel("Número de observações (país-ano)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Mostra onde se concentram os valores de expectativa de vida, com média e mediana marcadas. "
                   "Observe se há uma cauda de valores mais baixos.")
    with g2:
        dispersao("indice_renda", "Índice de renda × expectativa de vida", "Índice de renda (0 a 1)")
        st.caption("Cada ponto é um país-ano do conjunto de treino. O índice de renda varia de 0 a 1; observa-se uma tendência de "
                   "expectativa de vida mais alta onde o índice é maior. É uma associação descritiva, não indica que um fator causa o outro.")

    g3, g4 = st.columns(2)
    with g3:
        dispersao("escolaridade_anos", "Escolaridade média e expectativa de vida", "Escolaridade média (anos)")
        st.caption("Cada ponto é um país-ano. Há uma tendência de maior expectativa de vida onde a escolaridade média é maior. "
                   "É uma associação, não uma relação de causa e efeito.")
    with g4:
        dispersao("pib_per_capita", "PIB per capita e expectativa de vida", "PIB per capita (US$, escala logarítmica)", log_x=True)
        st.caption("A escala logarítmica facilita a leitura de valores muito dispersos. Observe a tendência de maior expectativa de vida "
                   "em PIBs mais altos e se ela diminui nos níveis mais altos.")

    st.subheader("Amostra da base")
    n_linhas = st.slider("Quantidade de linhas exibidas", 5, 100, 10)
    st.dataframe(df.head(n_linhas), hide_index=True)
    st.download_button("Baixar a base de treino filtrada (CSV)", data=para_csv(df), file_name="base_treino_filtrada.csv", mime="text/csv")

    st.subheader("Estatísticas descritivas")
    cols_desc = [ALVO] + [c for c in NUM if c in df.columns]
    st.dataframe(df[cols_desc].rename(columns=ROT).describe().T.round(2))

# ============================================================================
# ABA 3 — Diagnóstico do modelo (resultados lidos do notebook; nada é recalculado)
# ============================================================================
with aba_diag:
    st.subheader("Métricas finais")
    st.caption("Resultados obtidos no conjunto de teste isolado, avaliado uma única vez.")
    d1, d2, d3 = st.columns(3)
    with d1.container(border=True):
        st.metric("RMSE", f"{fmt(m_teste['RMSE'])} anos")
    with d2.container(border=True):
        st.metric("MAE", f"{fmt(m_teste['MAE'])} anos")
    with d3.container(border=True):
        st.metric("R²", fmt(m_teste["R2"], 3))

    with st.expander("Como o modelo final foi escolhido"):
        st.markdown(
            "A configuração promovida ao teste foi selecionada pela **regra previamente definida no protocolo**. "
            "Primeiro são consideradas as configurações cujo RMSE de validação está dentro do limite de 1 erro-padrão "
            "em relação à melhor média de validação cruzada. Entre as configurações elegíveis, a seleção considera o menor gap "
            "entre treino e validação e, em caso de empate, o menor tempo de ajuste.\n\n"
            "Essa regra é aplicada usando somente treino/validação; o teste não participa dessa escolha. "
            "Grid Search e Optuna também usaram apenas os dados de desenvolvimento (treino), com os mesmos folds por país. "
            "Como esses folds orientaram a busca e a escolha, o RMSE de validação pós-tuning não é uma estimativa "
            "perfeitamente independente; a estimativa final de generalização é a do teste. Não foi usada validação cruzada aninhada.")

    st.subheader("Comparação das 9 configurações")
    tab = comparacao.copy()
    partes = tab["configuração"].str.split(" — ", n=1, expand=True)
    tab["Modelo"], tab["Estratégia"] = partes[0], partes[1]
    tab["Situação"] = np.where(tab["selecionada"], "Selecionada",
                               np.where(tab["elegivel_1se"], "Elegível (1 erro-padrão)", ""))
    exibir = tab[["Modelo", "Estratégia", "RMSE CV (média)", "RMSE CV (dp)", "RMSE treino", "gap (CV−treino)", "Situação"]].copy()
    exibir.columns = ["Modelo", "Estratégia", "RMSE de validação", "Desvio entre folds", "RMSE de treino", "Gap treino-validação", "Situação"]
    st.dataframe(exibir.style.format({c: "{:.3f}" for c in exibir.columns[2:6]})
                 .apply(lambda linha: ["font-weight: bold; background-color: #d8f0d8; color: #111" if linha["Situação"] == "Selecionada" else ""
                                       for _ in linha], axis=1),
                 hide_index=True)
    st.caption("RMSE em anos. Gap = RMSE de validação − RMSE de treino. A linha em verde é a configuração final.")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(tab))[::-1]
    ax.barh(y, tab["RMSE CV (média)"], xerr=tab["RMSE CV (dp)"], capsize=3,
            color=["seagreen" if s else "lightsteelblue" for s in tab["selecionada"]])
    ax.set_yticks(y)
    ax.set_yticklabels(tab["configuração"])
    ax.set_xlabel("RMSE de validação cruzada (anos, média ± desvio entre folds)")
    ax.set_title("Comparação das configurações por RMSE de validação")
    ax.legend(handles=[Patch(color="seagreen", label="Configuração selecionada"), Patch(color="lightsteelblue", label="Demais configurações")])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Quanto menor o RMSE, menor foi o erro médio observado durante a validação. Barras com desvios sobrepostos indicam "
               "desempenho de validação semelhante, por isso a escolha seguiu uma regra definida antes do teste.")

    st.subheader("Valores reais × valores previstos")
    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(pred_teste["y_real"], pred_teste["y_previsto"], s=14, alpha=0.5, label="Observações do teste")
        lim = [pred_teste[["y_real", "y_previsto"]].min().min() - 1, pred_teste[["y_real", "y_previsto"]].max().max() + 1]
        ax.plot(lim, lim, "r--", label="Previsão perfeita (y = x)")
        ax.set_xlabel("Expectativa de vida real (anos)")
        ax.set_ylabel("Expectativa de vida prevista (anos)")
        ax.set_title("Real × previsto (conjunto de teste)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Quanto mais próximos os pontos estiverem da linha de referência, mais próxima a previsão está do valor observado.")
    with cb:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(pred_teste["y_previsto"], pred_teste["residuo"], s=14, alpha=0.5, color="darkorange", label="Observações do teste")
        ax.axhline(0, color="red", ls="--", label="Resíduo zero")
        ax.set_xlabel("Expectativa de vida prevista (anos)")
        ax.set_ylabel("Resíduo: real − previsto (anos)")
        ax.set_title("Previsão × resíduo (conjunto de teste)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Resíduo positivo indica subestimação e resíduo negativo indica superestimação. "
                   "Pontos espalhados em torno de zero, sem padrão evidente, sugerem ausência de viés sistemático.")

    st.subheader("Importância das variáveis")
    ordenado = imp_perm.sort_values("importancia_media")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordenado["rotulo"], ordenado["importancia_media"], xerr=ordenado["desvio_entre_folds"], capsize=2, color="slateblue",
            label="Média entre folds (± desvio)")
    ax.set_xlabel("Aumento do RMSE ao embaralhar a variável (anos)")
    ax.set_title("Importância por permutação (configuração final)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("A importância por permutação mostra o quanto o desempenho do modelo se altera quando uma variável é embaralhada. "
               "Ela descreve a dependência do modelo e não representa causalidade.")

    with st.expander("Verificação técnica: Notebook × Streamlit"):
        casos = meta.get("casos_paridade") or [meta["caso_paridade"]]
        tol = meta.get("tolerancia_paridade", {"rtol": 1e-9, "atol": 1e-6})
        # Entradas exatamente como salvas pelo notebook (None vira NaN; o Pipeline carregado imputa e codifica)
        X_par = pd.DataFrame([c["entrada"] for c in casos])[FEATURES].astype({c: float for c in NUM})
        pred_app = modelo_final.predict(X_par)                       # mesmo Pipeline carregado de modelo_final.joblib
        esperado = np.array([c["previsao_notebook"] for c in casos])
        difs = np.abs(pred_app - esperado)
        tab_par = pd.DataFrame({"País": [c["pais"] for c in casos], "Ano": [c["ano"] for c in casos],
                                "Valor real": [c["y_real"] for c in casos], "Previsão no notebook": esperado,
                                "Previsão no app": pred_app, "Diferença absoluta": difs})
        st.dataframe(tab_par.style.format({"Valor real": "{:.1f}", "Previsão no notebook": "{:.6f}", "Previsão no app": "{:.6f}", "Diferença absoluta": "{:.2e}"}),
                     hide_index=True)
        if np.allclose(pred_app, esperado, rtol=tol["rtol"], atol=tol["atol"]):
            st.success(f"Paridade confirmada em {len(casos)} casos (diferença máxima {difs.max():.2e}): o app usa o mesmo Pipeline avaliado no notebook.")
        else:
            st.error(f"Divergência entre o app e o notebook (diferença máxima {difs.max():.2e}). Investigar antes da entrega.")
        st.caption(f"Comparação numérica com tolerância de ponto flutuante (rtol = {tol['rtol']:g}, atol = {tol['atol']:g}), pois uma mesma previsão "
                   "pode diferir por arredondamento. Esta verificação não altera o modelo.")

# ============================================================================
# ABA 4 — Fazer uma previsão
# ============================================================================
with aba_prev:
    st.subheader("Calcular a expectativa de vida")
    st.markdown("Informe as características de um país-ano. Os campos já vêm preenchidos com valores típicos "
                "(mediana do treino); altere apenas o que quiser testar.")

    GRUPOS = [
        ("País e período", ["ano", "status", "regiao"]),
        ("Economia e educação", ["pib_per_capita", "indice_renda", "escolaridade_anos"]),
        ("Saúde e gastos", ["alcool", "gasto_saude_pct_pib_pc", "gasto_saude_total_pct"]),
        ("Demografia e saúde infantil", ["populacao", "obitos_infantis", "magreza_1_19", "magreza_5_9"]),
        ("Vacinação (coberturas)", ["hepatite_b", "polio", "difteria"]),
    ]
    restantes = [c for c in FEATURES if c not in {v for _, vs in GRUPOS for v in vs}]
    if restantes:                                   # segurança: nenhuma variável do Pipeline pode ficar sem campo
        GRUPOS.append(("Outras variáveis", restantes))
    GRUPOS = [(t, [v for v in vs if v in FEATURES]) for t, vs in GRUPOS]

    def rotulo_campo(c):
        return ROT.get(c, c)

    # Valores iniciais (mediana do treino / categoria mais frequente), definidos uma única vez no session_state
    for c in NUM:
        med = meta["faixas_numericas"][c][2]
        st.session_state.setdefault(f"val_{c}", int(round(med)) if meta["tipos_numericos"][c] == "inteiro" else float(med))
    for c in CAT:
        st.session_state.setdefault(f"val_{c}", meta["categoria_padrao"][c])
    st.session_state.setdefault("ausentes", [])

    def preencher_caso():
        """Preenche o formulário com um caso do teste de consistência (para conferir a paridade com o notebook)."""
        ent = meta["caso_paridade"]["entrada"]
        for c in NUM:
            v = ent.get(c)
            usar = meta["faixas_numericas"][c][2] if v is None else v
            st.session_state[f"val_{c}"] = int(round(usar)) if meta["tipos_numericos"][c] == "inteiro" else float(usar)
        for c in CAT:
            st.session_state[f"val_{c}"] = ent[c]
        st.session_state["ausentes"] = [c for c in NUM if ent.get(c) is None]

    cp = meta["caso_paridade"]
    st.button(f"Preencher com o caso do teste de consistência: {cp['pais']} ({cp['ano']})", on_click=preencher_caso)

    valores, fora = {}, []
    with st.form("form_previsao"):
        for k, (titulo, vars_grupo) in enumerate(GRUPOS):
            with st.expander(titulo, expanded=(k < 2)):
                cols = st.columns(2)
                for j, c in enumerate(vars_grupo):
                    with cols[j % 2]:
                        v = info_var.get(c, {"descricao": c, "unidade": ""})
                        if c in NUM:
                            lo, hi, med = meta["faixas_numericas"][c]
                            inteiro = meta["tipos_numericos"][c] == "inteiro"
                            if inteiro:
                                passo = max(1, int(10 ** np.floor(np.log10(max(hi - lo, 1) / 50))))
                                kw = dict(min_value=0, step=passo, format="%d")
                            else:
                                passo = float(10 ** np.floor(np.log10(max(hi - lo, 1e-9) / 50)))
                                kw = dict(min_value=0.0, step=passo, format="%.10g")
                            valores[c] = st.number_input(
                                rotulo_campo(c), key=f"val_{c}",
                                help=f"{v['descricao']}. Unidade: {v['unidade']}. Intervalo observado no treino: {lo:g} a {hi:g}.", **kw)
                            st.caption(f"Observado no treino: {lo:g} a {hi:g}")
                        else:
                            opcoes = meta["categorias"][c]
                            valores[c] = st.selectbox(
                                rotulo_campo(c), opcoes, key=f"val_{c}",
                                format_func=lambda x, c=c: STATUS_PT.get(x, x) if c == "status" else x,
                                help=v["descricao"])
        with st.expander("Opções avançadas — valores ausentes"):
            st.caption("Se uma informação não estiver disponível, marque-a aqui. O valor será enviado como ausente "
                       "e o Pipeline salvo fará a imputação (o app não recria essa etapa).")
            ausentes = st.multiselect("Variáveis não informadas", NUM, format_func=rotulo_campo, key="ausentes")
        enviado = st.form_submit_button("Calcular expectativa de vida", type="primary")

    if enviado:
        for c in ausentes:
            valores[c] = np.nan
        for c in NUM:
            lo, hi, _ = meta["faixas_numericas"][c]
            if c not in ausentes and (valores[c] < lo or valores[c] > hi):
                fora.append(rotulo_campo(c))
        # DataFrame com EXATAMENTE as colunas do Pipeline final, na mesma ordem; sem pré-processamento no app
        df_entrada = pd.DataFrame([valores])[FEATURES].astype({c: float for c in NUM})
        previsao = float(modelo_final.predict(df_entrada)[0])

        with st.container(border=True):
            st.markdown("**Expectativa de vida estimada**")
            st.markdown(f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>{fmt(previsao)} anos</div>", unsafe_allow_html=True)
            st.caption(f"É uma estimativa do modelo ({MODELO_NOME}, {ESTRATEGIA}), não um valor oficial nem uma relação causal. "
                       f"Em países inéditos do teste, o erro absoluto médio foi de {fmt(m_teste['MAE'], 1)} anos.")
        if fora:
            st.warning("Este valor está fora do intervalo observado nos dados de treinamento. A previsão representa uma extrapolação "
                       "e deve ser interpretada com cautela. Variáveis fora do intervalo: " + ", ".join(fora) + ".")
        # Paridade pela interface: se o formulário contém exatamente o caso salvo pelo notebook, compara as previsões
        ent = cp["entrada"]
        def _igual(c):
            v = ent.get(c)
            if c in NUM:
                return (v is None and c in ausentes) or (v is not None and c not in ausentes and np.isclose(valores[c], v, rtol=1e-9, atol=1e-9))
            return valores[c] == v
        if all(_igual(c) for c in FEATURES):
            tol = meta.get("tolerancia_paridade", {"rtol": 1e-9, "atol": 1e-6})
            dif = abs(previsao - cp["previsao_notebook"])
            if np.isclose(previsao, cp["previsao_notebook"], rtol=tol["rtol"], atol=tol["atol"]):
                st.success(f"Paridade com o notebook ({cp['pais']}, {cp['ano']}): notebook {cp['previsao_notebook']:.6f} · aplicativo {previsao:.6f} · diferença {dif:.1e}.")
            else:
                st.error(f"Divergência com o notebook ({cp['pais']}, {cp['ano']}): diferença {dif:.1e}. Investigar antes da entrega.")
        with st.expander("Dados enviados ao Pipeline"):
            st.dataframe(df_entrada, hide_index=True)

st.caption("Aplicação do Checkpoint 5 (FIAP). Dados: Life Expectancy (WHO), Kaggle. Resultados lidos dos arquivos gerados pelo notebook.")