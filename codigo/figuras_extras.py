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
    B = inst.B
    nb = np.bincount(inst.predio, minlength=B)
    # posições aproximadas, como no mapa (SEDE e Anexo I juntos; III e IV do
    # outro lado da Av. Pontes Vieira)
    pos = {0: (0.0, 1.05), 1: (0.15, 0.05), 2: (1.45, 0.55), 3: (1.1, -1.25), 4: (2.75, -0.85)}
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.plot([-0.6, 3.9], [-0.6, -0.05], color="#9a9a92", linewidth=6, alpha=0.35, zorder=0)
    ax.text(3.85, 0.08, "Av. Pontes Vieira", fontsize=7, color="#55554f", ha="right", style="italic")
    for b, (x, y) in pos.items():
        ax.scatter([x], [y], s=1700, color=PREDIO_COR[b], zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, y, PREDIO_NOME[b].replace("Anexo ", "A"), ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", zorder=4)
        ax.text(x + 0.4, y, f"{nb[b]} aparelhos\n{inst.cap[b]} vagas/dia", ha="left", va="center",
                fontsize=7, color="#3d4556", zorder=4)
    ax.set_xlim(-0.6, 3.9)
    ax.set_ylim(-1.8, 1.6)
    ax.axis("off")
    ax.set_title(f"{inst.n} aparelhos · {inst.m} equipes de 2 técnicos · {inst.H} min/dia", fontsize=8.5)
    nomes = [n.replace("Anexo ", "A") for n in PREDIO_NOME[:B]]
    tab = axt.table(cellText=[[str(v) for v in linha] for linha in inst.T], rowLabels=nomes, colLabels=nomes,
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.5)
    for (r, c), cel in tab.get_celld().items():
        cel.set_edgecolor("#c3c2b7")
        if r == 0 or c == -1:
            cel.set_text_props(fontweight="bold", color="#14213d")
        elif r - 1 == c:
            cel.set_facecolor("#ecebe5")
    axt.axis("off")
    axt.set_title("Deslocamento (min); diagonal: troca de sala/andar", fontsize=8.5)
    fig.tight_layout()
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
    fig.legend(handles, PREDIO_NOME + ["Deslocamento"], loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
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
    # As duas figuras abaixo dependem da validação exata (modelo_exato.py),
    # rodada à parte; são puladas se ela ainda não existir.
    if os.path.exists(os.path.join(BASE, "resultados", "validacao_exata.csv")):
        exemplo_decodificador()
        tempo_milp()
    else:
        print("validacao_exata.csv não encontrado: figuras do MILP puladas")
