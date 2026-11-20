"""
Figuras de apoio para o relatório e a apresentação:
  1. esquema do problema (prédios, deslocamentos, aparelhos e vagas);
  2. exemplo pequeno do decodificador (permutação -> programação);
  3. tempo do MILP (CBC) por tamanho de instância.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmoc import Instancia, decodificar
from gantt import programa, PREDIO_COR, PREDIO_NOME
from metaheuristicas import vns
from estilo import aplicar_estilo, CINZA

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = os.path.join(BASE, "figuras")


def esquema():
    inst = Instancia(os.path.join(BASE, "instancias", "ALECE-base.txt"))
    nb = np.bincount(inst.predio, minlength=4)
    pos = {0: (0.0, 0.8), 1: (2.6, 0.8), 2: (2.6, -0.8), 3: (0.0, -0.8)}
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for a in range(4):
        for b in range(a + 1, 4):
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], color="#c3c2b7", linewidth=1.5, zorder=1)
            diagonal = (x1 != x2) and (y1 != y2)
            f = 0.28 if diagonal else 0.5   # rótulos das diagonais fora do cruzamento
            ax.text(x1 + f * (x2 - x1), y1 + f * (y2 - y1), f"{inst.T[a, b]} min", fontsize=8, ha="center",
                    va="center", color="#3d4556", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none"))
    for b, (x, y) in pos.items():
        ax.scatter([x], [y], s=2600, color=PREDIO_COR[b], zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, y, PREDIO_NOME[b], ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", zorder=4)
        dy = 0.42 if y > 0 else -0.42
        ax.text(x, y + dy, f"{nb[b]} aparelhos · {inst.cap[b]} vagas/dia", ha="center",
                va="bottom" if y > 0 else "top", fontsize=7.5, color="#3d4556", zorder=4)
    ax.text(1.3, -1.62, f"{inst.m} equipes de 2 técnicos · jornada de {inst.H} min/dia · cada dia começa na SEDE · "
                        f"troca de sala no mesmo prédio: {inst.T[0, 0]} min",
            fontsize=7.5, color="#3d4556", ha="center", va="top")
    ax.set_xlim(-0.8, 3.4)
    ax.set_ylim(-1.85, 1.45)
    ax.axis("off")
    fig.savefig(os.path.join(F, "esquema_problema.pdf"))
    fig.savefig(os.path.join(F, "esquema_problema.png"), dpi=200)


def gantt_pequeno(ax, inst, df, titulo):
    for _, r in df.iterrows():
        x0 = r.dia * inst.H
        ax.barh(r.equipe, r.desloc, left=x0 + r.inicio, height=0.55, color="#c3c2b7", edgecolor="none")
        ax.barh(r.equipe, r.fim - r.chegada, left=x0 + r.chegada, height=0.55,
                color=PREDIO_COR[r.predio], edgecolor="white", linewidth=0.8)
        ax.text(x0 + (r.chegada + r.fim) / 2, r.equipe, str(r.aparelho + 1), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.axvline(inst.H, color="#0b0b0b", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(inst.m))
    ax.set_yticklabels([f"Equipe {k + 1}" for k in range(inst.m)])
    ax.invert_yaxis()
    ax.set_title(titulo, fontsize=8.5)
    ax.grid(False)


def exemplo_decodificador():
    inst = Instancia(os.path.join(BASE, "instancias_pequenas", "PEQ-n8-m2-1.txt"))
    args = inst.args()
    ordem = np.argsort(-inst.p, kind="stable").astype(np.int64)  # ordem LPT
    s_opt, _ = vns(inst, kmax=3, fator_L=2.0, tempo=1.0, semente=0)
    c1 = decodificar(ordem, inst.n, *args)
    c2 = decodificar(s_opt, inst.n, *args)
    val = pd.read_csv(os.path.join(BASE, "resultados", "validacao_exata.csv")).set_index("instancia")
    otimo = int(round(val.loc["PEQ-n8-m2-1", "cmax_milp"]))
    fmt = lambda s: "(" + ", ".join(str(j + 1) for j in s) + ")"
    fig, eixos = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)
    gantt_pequeno(eixos[0], inst, programa(inst, ordem), f"π = {fmt(ordem)}  →  Cmax = {c1[0]} min")
    gantt_pequeno(eixos[1], inst, programa(inst, s_opt),
                  f"π = {fmt(s_opt)}  →  Cmax = {c2[0]} min  (ótimo do MILP: {otimo} min)")
    eixos[1].set_xlabel("Minutos do dia (jornada H = 420 min)")
    eixos[1].set_xlim(0, inst.H + 10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PREDIO_COR] + [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")]
    fig.legend(handles, PREDIO_NOME + ["Deslocamento"], loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(F, "exemplo_decodificador.pdf"))
    fig.savefig(os.path.join(F, "exemplo_decodificador.png"), dpi=200)
    print("exemplo:", c1[0], c2[0], "ótimo", otimo)


def tempo_milp():
    val = pd.read_csv(os.path.join(BASE, "resultados", "validacao_exata.csv"))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for m, mk in [(1, "o"), (2, "s")]:
        d = val[val.m == m]
        ax.scatter(d.n, d.tempo_milp, marker=mk, s=28, color="#2a78d6" if m == 1 else "#eb6834",
                   label=f"{m} equipe" + ("s" if m > 1 else ""), zorder=3)
    med = val.groupby("n").tempo_milp.mean()
    ax.plot(med.index, med.values, color=CINZA, linewidth=1.5, zorder=2, label="Média")
    ax.set_yscale("log")
    ax.set_xticks(sorted(val.n.unique()))
    ax.set_xlabel("Número de aparelhos (n)")
    ax.set_ylabel("Tempo do CBC até o ótimo (s)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(F, "tempo_milp.pdf"))
    fig.savefig(os.path.join(F, "tempo_milp.png"), dpi=200)


if __name__ == "__main__":
    aplicar_estilo()
    os.makedirs(F, exist_ok=True)
    esquema()
    exemplo_decodificador()
    tempo_milp()
