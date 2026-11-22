"""
Boxplot da melhoria de cada execução sobre a solução inicial LPT+NEH e
convergência por iteração no Caso-base (um painel, eixo x em escala log,
porque o número de iterações varia muito entre os algoritmos).

Gera figuras/boxplot_melhoria.{pdf,png} e figuras/convergencia_caso.{pdf,png}.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estilo import aplicar_estilo, COR

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(BASE, "resultados")
F = os.path.join(BASE, "figuras")
ALGS = ["IG", "VNS", "ILS", "AG"]


def main():
    aplicar_estilo()
    ex = pd.read_csv(os.path.join(R, "experimento_execucoes.csv"))
    det = pd.read_csv(os.path.join(R, "deterministicos.csv"))
    neh = det[det.algoritmo == "LPT+NEH"].set_index("instancia").cmax
    ex["melhoria"] = 100 * (ex.instancia.map(neh) - ex.cmax) / ex.instancia.map(neh)

    # ---- boxplot da melhoria sobre a LPT+NEH (18 instâncias x 30 execuções)
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    dados = [ex[ex.algoritmo == a].melhoria.values for a in ALGS]
    bp = ax.boxplot(dados, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=2.5, alpha=0.5, mec="none"))
    for caixa, fl, a in zip(bp["boxes"], bp["fliers"], ALGS):
        caixa.set_facecolor(COR[a]); caixa.set_edgecolor("#55554f")
        fl.set_markerfacecolor(COR[a])
    for i, d in enumerate(dados, start=1):
        ax.plot(i, d.mean(), marker="D", ms=3.5, color="white", mec="black", mew=0.7, zorder=5)
    ax.set_xticks(range(1, 5), ALGS)
    ax.set_ylabel("Melhoria sobre a LPT+NEH (%)")
    ax.axhline(0, color="#8a8a83", lw=0.8, zorder=0)
    fig.savefig(os.path.join(F, "boxplot_melhoria.pdf"))
    fig.savefig(os.path.join(F, "boxplot_melhoria.png"), dpi=200)
    plt.close(fig)

    # ---- convergência por iteração no Caso-base (mediana e IQR das 30 execuções)
    cv = pd.read_csv(os.path.join(R, "convergencia.csv"))
    cv = cv[(cv.instancia == "Caso-base") & cv.iteracao.notna()]
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for a in ALGS:
        c = cv[cv.algoritmo == a]
        itmax = int(c.iteracao.max())
        grade = np.unique(np.geomspace(1, max(itmax, 2), 200).astype(int))
        curvas = []
        for _, g in c.groupby("rep"):
            g = g.sort_values("iteracao")
            idx = np.searchsorted(g.iteracao.values, grade, side="right") - 1
            curvas.append(g.f.values[np.clip(idx, 0, None)])
        q1, med, q3 = np.percentile(curvas, [25, 50, 75], axis=0)
        ax.fill_between(grade, q1, q3, color=COR[a], alpha=0.18, lw=0)
        ax.plot(grade, med, color=COR[a], lw=1.6, label=a)
    ax.set_xscale("log")
    ax.set_xlabel("Iteração (geração, no AG) — escala log")
    ax.set_ylabel("Melhor Cmax (min)")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(os.path.join(F, "convergencia_caso.pdf"))
    fig.savefig(os.path.join(F, "convergencia_caso.png"), dpi=200)
    plt.close(fig)

    print(ex.groupby("algoritmo").melhoria.describe().round(3))


if __name__ == "__main__":
    main()
