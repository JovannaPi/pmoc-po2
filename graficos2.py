import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd

B = "/home/claude/pmoc-po2/"
sys.path.insert(0, B + "codigo")
from pmoc import Instancia, decodificar
from gantt import programa

ROXO = "#522276"; MEDIO = "#885da8"; LILAS = "#a88ae1"; CLARO = "#e2d1ed"; CINZA = "#9a8fa6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": CINZA,
                     "axes.labelcolor": ROXO, "xtick.color": ROXO, "ytick.color": ROXO,
                     "axes.spines.top": False, "axes.spines.right": False})


def cartao(f):
    f.patches.append(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0,rounding_size=0.035",
                                    transform=f.transFigure, fc=CLARO, ec="none", zorder=-10))


# ---- Gantt do dia 1 (melhor solução do VNS na ALECE-base)
inst = Instancia(B + "instancias/ALECE-base.txt")
melhor = None
for r in range(30):
    s = np.load(B + f"resultados/solucoes/ALECE-base_VNS_{r}.npy")
    c = decodificar(s, inst.n, *inst.args())[0]
    if melhor is None or c < melhor[0]:
        melhor = (c, s)
df = programa(inst, melhor[1])
d1 = df[df.dia == 0]
CORES = ["#522276", "#2a78d6", "#1baf7a", "#eb6834", "#c0569b"]
NOMES = ["SEDE", "Anexo I", "Anexo II", "Anexo III", "Anexo IV"]
f = plt.figure(figsize=(6.6, 6), dpi=200); cartao(f)
ax = f.add_axes([0.17, 0.24, 0.78, 0.62]); ax.set_facecolor("none")
for _, r in d1.iterrows():
    ax.barh(r.equipe, r.desloc, left=r.inicio, height=0.62, color="#bfb6c9")
    ax.barh(r.equipe, r.fim - r.chegada, left=r.chegada, height=0.62, color=CORES[r.predio], edgecolor="white", lw=1.5)
ax.set_yticks(range(inst.m), [f"Equipe {k+1}" for k in range(inst.m)]); ax.invert_yaxis()
ax.set_xticks([0, 60, 120, 180, 240, 300, 360, 420], ["9h", "10h", "11h", "12h", "14h", "15h", "16h", "17h"]); ax.set_xlabel("Jornada útil (almoço das 12h às 13h fora do eixo)", fontsize=11.5)
ax.set_xlim(0, 420); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color="white", lw=1.5); ax.set_axisbelow(True)
leg = [Patch(color=CORES[i], label=NOMES[i]) for i in range(5)] + [Patch(color="#bfb6c9", label="Deslocamento")]
f.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=12.5, labelcolor=ROXO)
f.text(0.5, 0.92, "Agenda do 1º dia (proposta)", ha="center", color=ROXO, fontweight="bold", fontsize=17)
f.savefig("res_gantt.png", transparent=True); plt.close(f)

# ---- Convergência na ALECE-base (mediana das 30 execuções, por tempo)
cv = pd.read_csv(B + "resultados/convergencia.csv")
cv = cv[cv.instancia == "ALECE-base"]
grade = np.linspace(0, 21, 300)
f = plt.figure(figsize=(7, 5), dpi=200); cartao(f)
ax = f.add_axes([0.17, 0.16, 0.76, 0.68]); ax.set_facecolor("none")
med = {}
for alg in ["IG", "ILS", "AG", "VNS"]:
    curvas = []
    for _, g in cv[cv.algoritmo == alg].groupby("rep"):
        g = g.sort_values("tempo")
        idx = np.searchsorted(g.tempo.values, grade, side="right") - 1
        curvas.append(g.f.values[np.clip(idx, 0, None)])
    med[alg] = np.median(curvas, axis=0)
ax.plot(grade, med["IG"], color=LILAS, lw=5, label="IG, ILS e AG (mediana)")
ax.plot(grade, med["VNS"], color=ROXO, lw=4, label="VNS (mediana)")
ax.set_xlabel("Tempo de execução (s)"); ax.set_ylabel("Melhor Cmax (min)")
ax.grid(axis="y", color="white", lw=1.5); ax.legend(frameon=False, labelcolor=ROXO, loc="upper right", bbox_to_anchor=(1.0, 0.93))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
f.text(0.5, 0.9, "Convergência na ALECE-base", ha="center", color=ROXO, fontweight="bold", fontsize=17)
f.savefig("res_conv.png", transparent=True); plt.close(f)
print(melhor[0], len(d1), {a: (v[0], v[-1]) for a, v in med.items()})
