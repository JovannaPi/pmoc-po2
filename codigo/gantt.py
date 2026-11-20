"""
Gráfico de Gantt (equipes x dias) da melhor solução encontrada para a
instância ALECE-base, comparado com a execução real da regra atual.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pmoc import Instancia, regra_atual, decodificar
from metaheuristicas import ig, vns, ils
from estilo import aplicar_estilo

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PREDIO_COR = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c0569b"]  # SEDE, A1..A4
PREDIO_NOME = ["SEDE", "Anexo I", "Anexo II", "Anexo III", "Anexo IV"]


def programa(inst, seq):
    """Mesma lógica do decodificador, devolvendo a programação detalhada."""
    m, H = inst.m, inst.H
    dia, t, loc = [0] * m, [0] * m, [0] * m
    uso = np.zeros((inst.maxD, inst.B), int)
    tarefas = []
    for j in seq:
        b = inst.predio[j]
        melhor = None
        for k in range(m):
            d = dia[k]
            des = inst.T[loc[k], b]
            ini = t[k] + des
            fim = ini + inst.p[j]
            if not (fim <= H and uso[d, b] < inst.cap[b]):
                d += 1
                while uso[d, b] >= inst.cap[b]:
                    d += 1
                des = inst.T[0, b]
                ini = des
                fim = des + inst.p[j]
            ab = d * H + fim
            if melhor is None or ab < melhor[0]:
                melhor = (ab, k, d, ini, fim, des)
        _, k, d, ini, fim, des = melhor
        dia[k], t[k], loc[k] = d, fim, b
        uso[d, b] += 1
        tarefas.append(dict(aparelho=int(j), predio=int(b), equipe=k, dia=d, inicio=ini - des,
                            chegada=ini, fim=fim, desloc=des))
    return pd.DataFrame(tarefas)


def desenhar(ax, df, H, titulo, ndias):
    for _, r in df.iterrows():
        x0 = r.dia * H
        ax.barh(r.equipe, r.desloc, left=x0 + r.inicio, height=0.6, color="#c3c2b7", edgecolor="none")
        ax.barh(r.equipe, r.fim - r.chegada, left=x0 + r.chegada, height=0.6,
                color=PREDIO_COR[r.predio], edgecolor="white", linewidth=0.6)
    for d in range(1, ndias + 1):
        ax.axvline(d * H, color="#0b0b0b", linewidth=0.8, linestyle="--")
    ax.set_xticks([d * H + H / 2 for d in range(ndias)])
    ax.set_xticklabels([f"Dia {d + 1}" for d in range(ndias)])
    ax.set_yticks(range(df.equipe.max() + 1))
    ax.set_yticklabels([f"Eq. {k + 1}" for k in range(df.equipe.max() + 1)])
    ax.invert_yaxis()
    ax.set_xlim(0, ndias * H)
    ax.set_title(titulo, fontsize=9)
    ax.grid(False)


def main():
    aplicar_estilo()
    inst = Instancia(os.path.join(BASE, "instancias", "ALECE-base.txt"))
    ex = pd.read_csv(os.path.join(BASE, "resultados", "experimento_execucoes.csv"))
    a = ex[ex.instancia == "ALECE-base"].sort_values("cmax").iloc[0]
    with open(os.path.join(BASE, "resultados", "parametros_finais.json")) as f:
        P = json.load(f)
    fun = {"IG": ig, "VNS": vns, "ILS": ils}[a.algoritmo]
    s, _ = fun(inst, tempo=0.02 * inst.n, semente=int(a.rep), **P[a.algoritmo])
    ra = regra_atual(inst)
    d_ra = programa(inst, ra)
    d_mh = programa(inst, s)
    c_ra = decodificar(ra, inst.n, *inst.args())
    c_mh = decodificar(s, inst.n, *inst.args())
    ndias = int(max(c_ra[3], c_mh[3]))
    fig, eixos = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    desenhar(eixos[0], d_ra, inst.H, f"Regra atual — {c_ra[3]} dias (Cmax = {c_ra[0]} min úteis, deslocamento = {c_ra[2]} min)", ndias)
    desenhar(eixos[1], d_mh, inst.H, f"{a.algoritmo} — {c_mh[3]} dias (Cmax = {c_mh[0]} min úteis, deslocamento = {c_mh[2]} min)", ndias)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PREDIO_COR] + [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")]
    fig.legend(handles, PREDIO_NOME + ["Deslocamento"], loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(BASE, "figuras", "gantt_alece.pdf"))
    fig.savefig(os.path.join(BASE, "figuras", "gantt_alece.png"), dpi=200)
    print("RA", c_ra, "MH", a.algoritmo, c_mh, "esperado", a.cmax)


if __name__ == "__main__":
    main()
