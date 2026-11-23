"""
Agenda do 1º dia do Caso-base na melhor solução do VNS (Gantt compacto
para o relatório). Usa as soluções salvas em resultados/solucoes.

Gera figuras/gantt_dia1.{pdf,png}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from estilo import aplicar_estilo
from pmoc import Instancia, decodificar
from gantt import programa

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c0569b"]
NOMES = ["SEDE", "Anexo I", "Anexo II", "Anexo III", "Anexo IV"]
DESLOC = "#c3c2b7"


def main():
    aplicar_estilo()
    inst = Instancia(os.path.join(BASE, "instancias", "Caso-base.txt"))
    melhor = None
    for r in range(30):
        s = np.load(os.path.join(BASE, "resultados", "solucoes", f"Caso-base_VNS_{r}.npy"))
        c = decodificar(s, inst.n, *inst.args())[0]
        if melhor is None or c < melhor[0]:
            melhor = (c, s)
    df = programa(inst, melhor[1])
    d1 = df[df.dia == 0]

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for _, r in d1.iterrows():
        ax.barh(r.equipe, r.desloc, left=r.inicio, height=0.65, color=DESLOC, edgecolor="none")
        ax.barh(r.equipe, r.fim - r.chegada, left=r.chegada, height=0.6,
                color=CORES[r.predio], edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(inst.m), [f"Eq. {k + 1}" for k in range(inst.m)])
    ax.invert_yaxis()
    ax.set_xlim(0, inst.H)
    ax.set_xticks(range(0, inst.H + 1, 60))
    ax.set_xlabel("Minutos úteis do dia")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    leg = [Patch(color=CORES[i], label=NOMES[i]) for i in range(5)] + [Patch(color=DESLOC, label="Deslocamento")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7.5,
              handlelength=1.0, columnspacing=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(BASE, "figuras", f"gantt_dia1.{ext}"), dpi=200)
    print("Cmax melhor VNS:", melhor[0], "| aparelhos no dia 1:", len(d1))


if __name__ == "__main__":
    main()
