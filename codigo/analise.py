"""
Análise do experimento final (metodologia da aula de Testes Estatísticos):
  - RPD de cada execução em relação à melhor solução conhecida da instância;
  - taxa de sucesso (execução com RPD <= 1%) e taxa de execuções que atingem
    o menor nº de dias conhecido;
  - normalidade (Shapiro-Wilk + Q-Q plot);
  - ANOVA + Tukey ou Kruskal-Wallis + Dunn, conforme a normalidade;
  - boxplot do RPD e curva de convergência (iterações).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from estilo import aplicar_estilo, COR, CINZA

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(BASE, "resultados")
F = os.path.join(BASE, "figuras")
ALGS = ["IG", "VNS", "ILS", "AG"]
H = 420


def main():
    aplicar_estilo()
    ex = pd.read_csv(os.path.join(R, "experimento_execucoes.csv"))
    det = pd.read_csv(os.path.join(R, "deterministicos.csv"))
    melhor = pd.concat([ex[["instancia", "cmax"]], det[["instancia", "cmax"]]]).groupby("instancia").cmax.min()
    melhor_dias = pd.concat([ex[["instancia", "dias"]], det[["instancia", "dias"]]]).groupby("instancia").dias.min()
    ex["rpd"] = (ex.cmax - ex.instancia.map(melhor)) / ex.instancia.map(melhor) * 100
    # Taxa de sucesso: execução a no máximo 1% da melhor solução conhecida
    ex["sucesso"] = ex.rpd <= 1.0
    ex["sucesso_dias"] = ex.dias == ex.instancia.map(melhor_dias)
    det["rpd"] = (det.cmax - det.instancia.map(melhor)) / det.instancia.map(melhor) * 100
    ex.to_csv(os.path.join(R, "experimento_rpd.csv"), index=False)

    out = []
    # ---------------- tabela por porte (n) e algoritmo
    tab = ex.groupby(["n", "algoritmo"]).agg(rpd=("rpd", "mean"), sucesso=("sucesso", "mean")).unstack()
    tdet = det.pivot_table(index="n", columns="algoritmo", values="rpd", aggfunc="mean")
    tab_final = pd.DataFrame(index=sorted(ex.n.unique()))
    for a in ["Regra atual", "LPT+NEH"]:
        tab_final[f"RPD {a}"] = tdet[a]
    for a in ALGS:
        tab_final[f"RPD {a}"] = tab[("rpd", a)]
    for a in ALGS:
        tab_final[f"TS {a}"] = tab[("sucesso", a)] * 100
    geral = {f"RPD {a}": det[det.algoritmo == a].rpd.mean() for a in ["Regra atual", "LPT+NEH"]}
    geral.update({f"RPD {a}": ex[ex.algoritmo == a].rpd.mean() for a in ALGS})
    geral.update({f"TS {a}": ex[ex.algoritmo == a].sucesso.mean() * 100 for a in ALGS})
    tab_final.loc["Média"] = geral
    for a in ALGS:
        tab_final[f"TSdias {a}"] = ex[ex.algoritmo == a].groupby("n").sucesso_dias.mean() * 100
        tab_final.loc["Média", f"TSdias {a}"] = ex[ex.algoritmo == a].sucesso_dias.mean() * 100
    tab_final.to_csv(os.path.join(R, "tabela_rpd_porte.csv"))
    out.append("== RPD médio (%) e taxa de sucesso (%) por porte\n" + tab_final.round(2).to_string())

    # ---------------- estatística
    grupos = [ex[ex.algoritmo == a].rpd.values for a in ALGS]
    sw = {a: stats.shapiro(g) for a, g in zip(ALGS, grupos)}
    out.append("== Shapiro-Wilk\n" + "\n".join(f"{a}: W={r.statistic:.4f} p={r.pvalue:.2e}" for a, r in sw.items()))
    normal = all(r.pvalue > 0.05 for r in sw.values())
    if normal:
        an = stats.f_oneway(*grupos)
        out.append(f"== ANOVA F={an.statistic:.3f} p={an.pvalue:.2e}")
        tk = pairwise_tukeyhsd(ex.rpd, ex.algoritmo)
        out.append(str(tk))
    else:
        kw = stats.kruskal(*grupos)
        out.append(f"== Kruskal-Wallis H={kw.statistic:.3f} p={kw.pvalue:.2e}")
        dunn = sp.posthoc_dunn(ex, val_col="rpd", group_col="algoritmo", p_adjust="bonferroni")
        out.append("== Dunn (Bonferroni)\n" + dunn.loc[ALGS, ALGS].to_string())
        dunn.to_csv(os.path.join(R, "dunn.csv"))
    # Teste de Friedman (blocos = instâncias): compara os algoritmos dentro de
    # cada instância, sem supor independência entre execuções da mesma
    # instância. Pós-teste de Nemenyi.
    bloco = ex.groupby(["instancia", "algoritmo"]).rpd.mean().unstack()[ALGS]
    fr = stats.friedmanchisquare(*[bloco[a].values for a in ALGS])
    ranks = bloco.rank(axis=1).mean()
    out.append(f"== Friedman (blocos = {len(bloco)} instâncias) Q={fr.statistic:.3f} p={fr.pvalue:.2e}\n"
               "Posto médio: " + ", ".join(f"{a}={ranks[a]:.2f}" for a in ALGS))
    nem = sp.posthoc_nemenyi_friedman(bloco.values)
    nem.index = nem.columns = ALGS
    out.append("== Nemenyi (pós-teste de Friedman)\n" + nem.to_string())
    nem.to_csv(os.path.join(R, "nemenyi.csv"))
    med = ex.groupby("algoritmo").rpd.agg(["mean", "median", "std"]).loc[ALGS]
    out.append("== Resumo RPD\n" + med.round(3).to_string())

    # ---------------- caso ALECE-base
    a = ex[ex.instancia == "ALECE-base"]
    d = det[det.instancia == "ALECE-base"]
    caso = [dict(metodo="Plano do sistema (modelo simplificado)", dias=int(d.dias_planejados.iloc[0]),
                 cmax=np.nan, deslocamento=np.nan)]
    for alg in ["Regra atual", "LPT+NEH"]:
        r = d[d.algoritmo == alg].iloc[0]
        caso.append(dict(metodo=alg, dias=r.dias, cmax=r.cmax, deslocamento=r.deslocamento))
    for alg in ALGS:
        r = a[a.algoritmo == alg]
        caso.append(dict(metodo=alg + " (média 30 exec.)", dias=r.dias.mean(), cmax=r.cmax.mean(),
                         deslocamento=r.deslocamento.mean(), melhor_cmax=r.cmax.min(),
                         sucesso=r.sucesso.mean() * 100, sucesso_dias=r.sucesso_dias.mean() * 100))
    caso = pd.DataFrame(caso)
    caso.to_csv(os.path.join(R, "caso_alece.csv"), index=False)
    out.append("== Caso ALECE-base (LB Cmax = %d min)\n" % d.lb.iloc[0] + caso.round(2).to_string())

    # Ganho médio frente à regra atual, em dias e minutos (todas as instâncias de teste)
    ra = det[det.algoritmo == "Regra atual"].set_index("instancia")
    melhor_alg = ex.groupby(["instancia", "algoritmo"])[["cmax", "dias"]].mean()
    ganhos = []
    for inst in ra.index:
        for alg in ALGS:
            r = melhor_alg.loc[(inst, alg)]
            ganhos.append(dict(instancia=inst, algoritmo=alg,
                               red_cmax=(ra.loc[inst, "cmax"] - r.cmax) / ra.loc[inst, "cmax"] * 100,
                               red_dias=ra.loc[inst, "dias"] - r.dias,
                               atraso_plano=ra.loc[inst, "dias"] - ra.loc[inst, "dias_planejados"]))
    ganhos = pd.DataFrame(ganhos)
    ganhos.to_csv(os.path.join(R, "ganhos_vs_regra_atual.csv"), index=False)
    out.append("== Redução média vs regra atual\n" + ganhos.groupby("algoritmo")[["red_cmax", "red_dias"]].mean().round(2).to_string()
               + f"\nDias a mais da regra atual em relação ao plano (média): {ganhos.atraso_plano.mean():.2f}")

    texto = "\n\n".join(out)
    with open(os.path.join(R, "analise_estatistica.txt"), "w") as f:
        f.write(texto)
    print(texto)

    # ---------------- figuras
    # Boxplot do RPD
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    bp = ax.boxplot(grupos, patch_artist=True, widths=0.55,
                    medianprops=dict(color="#0b0b0b", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=2.5, markeredgecolor=CINZA, alpha=0.6))
    for caixa, alg in zip(bp["boxes"], ALGS):
        caixa.set_facecolor(COR[alg])
        caixa.set_edgecolor("#55554f")
    ax.set_xticks(range(1, len(ALGS) + 1))
    ax.set_xticklabels(ALGS)
    ax.set_ylabel("RPD (%)")
    fig.savefig(os.path.join(F, "boxplot_rpd.pdf"))
    fig.savefig(os.path.join(F, "boxplot_rpd.png"), dpi=200)

    # Q-Q plots
    fig, eixos = plt.subplots(1, 4, figsize=(9, 2.4))
    for ax, alg, g in zip(eixos, ALGS, grupos):
        (osm, osr), (sl, ic, _) = stats.probplot(g)
        ax.plot(osm, osr, "o", markersize=2.5, color=COR[alg], alpha=0.7)
        ax.plot(osm, sl * osm + ic, color="#55554f", linewidth=1)
        ax.set_title(f"{alg} (Shapiro-Wilk p = {sw[alg].pvalue:.1e})", fontsize=8)
        ax.set_xlabel("Quantis teóricos")
    eixos[0].set_ylabel("Quantis do RPD")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "qqplot.pdf"))
    fig.savefig(os.path.join(F, "qqplot.png"), dpi=200)

    # Convergência (iterações): mediana entre as 30 execuções do melhor Cmax
    conv = pd.read_csv(os.path.join(R, "convergencia.csv"))
    fig, eixos = plt.subplots(1, 2, figsize=(5.4, 2.5))
    nomes_conv = ["ALECE-base"] + sorted(i for i in conv.instancia.unique() if i != "ALECE-base")
    for ax, inst in zip(eixos, nomes_conv):
        c = conv[conv.instancia == inst]
        tmax = c.tempo.max()
        grade = np.linspace(0, tmax, 200)
        for alg in ALGS:
            curvas = []
            for _, gr in c[c.algoritmo == alg].groupby("rep"):
                gr = gr.sort_values("tempo")
                idx = np.searchsorted(gr.tempo.values, grade, side="right") - 1
                curvas.append(gr.f.values[np.clip(idx, 0, None)])
            ax.step(grade, np.median(curvas, axis=0), where="post", color=COR[alg], linewidth=2, label=alg)
        ax.set_title(inst, fontsize=9)
        ax.set_xlabel("Tempo de CPU (s)")
        ax.legend(fontsize=7)
    eixos[0].set_ylabel("Melhor Cmax (min) — mediana")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "convergencia.pdf"))
    fig.savefig(os.path.join(F, "convergencia.png"), dpi=200)

    # Convergência por iteração (um painel por algoritmo, instância ALECE-base)
    if "iteracao" in conv.columns:
        c = conv[(conv.instancia == "ALECE-base") & conv.iteracao.notna()]
        nome_it = {"IG": "Iteração (destruição + construção)", "VNS": "Iteração (shaking + VND)",
                   "ILS": "Iteração (perturbação + busca local)", "AG": "Geração"}
        fig, eixos = plt.subplots(1, 4, figsize=(9, 2.4), sharey=True)
        for ax, alg in zip(eixos, ALGS):
            reps = list(c[c.algoritmo == alg].groupby("rep"))
            if not reps:
                continue
            itmax = max(int(g.iteracao.max()) for _, g in reps)
            grade = np.linspace(0, itmax, 300)
            curvas = []
            for _, g in reps:
                g = g.sort_values("iteracao")
                idx = np.searchsorted(g.iteracao.values, grade, side="right") - 1
                curvas.append(g.f.values[np.clip(idx, 0, None)])
            curvas = np.array(curvas)
            ax.fill_between(grade, np.percentile(curvas, 25, axis=0), np.percentile(curvas, 75, axis=0),
                            step="post", color=COR[alg], alpha=0.2, linewidth=0)
            ax.step(grade, np.median(curvas, axis=0), where="post", color=COR[alg], linewidth=2)
            ax.set_title(alg, fontsize=9)
            ax.set_xlabel(nome_it[alg], fontsize=7)
        eixos[0].set_ylabel("Melhor Cmax (min)")
        fig.suptitle("ALECE-base: mediana e intervalo interquartil das 30 execuções", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(F, "convergencia_iteracoes.pdf"))
        fig.savefig(os.path.join(F, "convergencia_iteracoes.png"), dpi=200)

    # RPD médio por tamanho de instância
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    g = ex[ex.instancia != "ALECE-base"].groupby(["n", "algoritmo"]).rpd.mean().unstack()
    for alg in ALGS:
        ax.plot(g.index, g[alg], marker="o", markersize=5, linewidth=2, color=COR[alg], label=alg)
    ax.set_xlabel("Número de aparelhos (n)")
    ax.set_ylabel("RPD médio (%)")
    ax.set_xticks(list(g.index))
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(os.path.join(F, "rpd_por_tamanho.pdf"))
    fig.savefig(os.path.join(F, "rpd_por_tamanho.png"), dpi=200)

    # Histórico do Optuna: desvio de cada tentativa e a melhor configuração da grade
    arq_t = os.path.join(R, "optuna_tentativas.csv")
    arq_c = os.path.join(R, "calibracao_comparacao.csv")
    if os.path.exists(arq_t) and os.path.exists(arq_c):
        tent = pd.read_csv(arq_t)
        comp = pd.read_csv(arq_c).set_index("algoritmo")
        fig, eixos = plt.subplots(1, 4, figsize=(9, 2.4))
        for ax, alg in zip(eixos, ALGS):
            t = tent[tent.algoritmo == alg].sort_values("tentativa")
            if t.empty:
                continue
            ax.plot(t.tentativa + 1, t.desvio, "o", color=COR[alg], markersize=4, alpha=0.6, label="Tentativa")
            ax.step(t.tentativa + 1, t.desvio.cummin(), where="post", color=COR[alg], linewidth=2, label="Melhor até aqui")
            if alg in comp.index:
                ax.axhline(comp.loc[alg, "desvio_grade"], color="#55554f", linestyle="--", linewidth=1, label="Melhor da grade")
            ax.set_title(alg, fontsize=9)
            ax.set_xlabel("Tentativa do Optuna")
        eixos[0].set_ylabel("Desvio (%)")
        eixos[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(F, "optuna_historico.pdf"))
        fig.savefig(os.path.join(F, "optuna_historico.png"), dpi=200)


if __name__ == "__main__":
    main()
