"""
Estudo de caso Caso-base: aparelhos concluídos ao fim de cada dia útil.
Compara o plano do sistema (modelo simplificado), a execução real da regra
atual (tempos reais + deslocamento) e a melhor solução das meta-heurísticas.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pmoc import Instancia, regra_atual, plano_regra_atual, decodificar
from gantt import programa
from estilo import aplicar_estilo, COR, CINZA

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def acumulado(dias, D):
    c = np.bincount(dias, minlength=D)[:D]
    return np.concatenate([[0], np.cumsum(c)])


def main():
    aplicar_estilo()
    inst = Instancia(os.path.join(BASE, "instancias", "Caso-base.txt"))
    ex = pd.read_csv(os.path.join(BASE, "resultados", "experimento_execucoes.csv"))
    a = ex[ex.instancia == "Caso-base"].sort_values(["cmax", "algoritmo", "rep"]).iloc[0]
    s = np.load(os.path.join(BASE, "resultados", "solucoes", f"Caso-base_{a.algoritmo}_{int(a.rep)}.npy"))
    c_mh = decodificar(s, inst.n, *inst.args())
    assert c_mh[0] == a.cmax
    ra = regra_atual(inst)
    c_ra = decodificar(ra, inst.n, *inst.args())
    d_plan = plano_regra_atual(inst)
    d_ra = programa(inst, ra).dia.values
    d_mh = programa(inst, s).dia.values
    D = int(max(d_ra.max(), d_mh.max(), d_plan.max())) + 1
    x = np.arange(D + 1)

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.step(x, acumulado(d_plan, D), where="post", color=CINZA, linewidth=2, linestyle="--",
            label=f"Plano do sistema ({d_plan.max() + 1} dias)")
    ax.step(x, acumulado(d_ra, D), where="post", color="#0b0b0b", linewidth=2,
            label=f"Regra atual, execução real ({c_ra[3]} dias)")
    ax.step(x, acumulado(d_mh, D), where="post", color=COR[a.algoritmo], linewidth=2,
            label=f"Melhor solução, {a.algoritmo} ({c_mh[3]} dias)")
    ax.axhline(inst.n, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel("Dia útil do ciclo")
    ax.set_ylabel("Aparelhos concluídos")
    ax.set_xlim(0, D)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, inst.n * 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(BASE, "figuras", "curva_conclusao_caso.pdf"))
    fig.savefig(os.path.join(BASE, "figuras", "curva_conclusao_caso.png"), dpi=200)
    pd.DataFrame(dict(dia=x, plano=acumulado(d_plan, D), regra_atual=acumulado(d_ra, D),
                      melhor=acumulado(d_mh, D))).to_csv(
        os.path.join(BASE, "resultados", "curva_conclusao_caso.csv"), index=False)
    # Gantt ampliado: os 2 primeiros dias do ciclo (o ciclo inteiro é ilegível)
    from gantt import desenhar, PREDIO_COR, PREDIO_NOME
    p_ra, p_mh = programa(inst, ra), programa(inst, s)
    fig, eixos = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    for ax, df, rot in [(eixos[0], p_ra, "Regra atual"), (eixos[1], p_mh, f"Melhor solução ({a.algoritmo})")]:
        dd = df[df.dia < 2]
        desenhar(ax, dd, inst.H, f"{rot} — dias 1 e 2: {len(dd)} aparelhos concluídos, "
                                 f"{int(dd.desloc.sum())} min de deslocamento", 2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PREDIO_COR] + [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")]
    fig.legend(handles, PREDIO_NOME + ["Deslocamento"], loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(BASE, "figuras", "gantt_caso_dias1e2.pdf"))
    fig.savefig(os.path.join(BASE, "figuras", "gantt_caso_dias1e2.png"), dpi=200)
    print("plano", d_plan.max() + 1, "RA", c_ra, "MH", a.algoritmo, c_mh)


if __name__ == "__main__":
    main()
