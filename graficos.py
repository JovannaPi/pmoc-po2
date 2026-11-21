import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

R = "/home/claude/pmoc-po2/resultados/"
ROXO = "#522276"; MEDIO = "#885da8"; LILAS = "#a88ae1"; CLARO = "#e2d1ed"; CINZA = "#9a8fa6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": CINZA,
                     "axes.labelcolor": ROXO, "xtick.color": ROXO, "ytick.color": ROXO,
                     "axes.spines.top": False, "axes.spines.right": False})


def cartao(f):
    f.patches.append(FancyBboxPatch((0.005, 0.01), 0.99, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                    transform=f.transFigure, fc=CLARO, ec="none", zorder=-10))


# Curva de conclusão
c = pd.read_csv(R + "curva_conclusao_alece.csv")
f = plt.figure(figsize=(8, 5), dpi=200); cartao(f)
ax = f.add_axes([0.12, 0.15, 0.84, 0.72]); ax.set_facecolor("none")
ax.plot(c.dia, c.plano, color=CINZA, lw=3, ls="--", label="Plano do sistema · 45 dias")
ax.plot(c.dia, c.regra_atual, color=LILAS, lw=4, label="Regra atual · 74 dias")
ax.plot(c.dia, c.melhor, color=ROXO, lw=4, label="Proposta · 69 dias")
ax.set_xlabel("Dia útil do ciclo"); ax.set_ylabel("Aparelhos concluídos")
ax.set_xlim(0, 75); ax.set_ylim(0, 1100); ax.grid(axis="y", color="white", lw=1.5)
ax.legend(frameon=False, loc="lower right", labelcolor=ROXO)
f.text(0.5, 0.92, "ALECE-base: 1050 aparelhos, 7 equipes", ha="center", color=ROXO, fontweight="bold", fontsize=17)
f.savefig("res_curva.png", transparent=True); plt.close(f)

# Sensibilidade
s = pd.read_csv(R + "sensibilidade_equipes.csv")
f = plt.figure(figsize=(8, 5), dpi=200); cartao(f)
ax = f.add_axes([0.12, 0.15, 0.84, 0.72]); ax.set_facecolor("none")
ax.axhline(45, color=CINZA, lw=3, ls="--")
ax.text(4.9, 46.5, "Plano do sistema: 45 dias", color=CINZA, ha="left", fontsize=14)
ax.plot(s.equipes, s.regra_atual, "o-", color=LILAS, lw=4, ms=10, label="Regra atual")
ax.plot(s.equipes, s.melhor_dias, "o-", color=ROXO, lw=4, ms=10, label="Proposta (VNS)")
for x, y in zip(s.equipes, s.melhor_dias):
    ax.text(x, y - 6.5, str(y), ha="center", color=ROXO, fontweight="bold", fontsize=14)
ax.set_xlabel("Número de equipes"); ax.set_ylabel("Dias do ciclo")
ax.set_xticks(s.equipes); ax.set_ylim(38, 108); ax.grid(axis="y", color="white", lw=1.5)
ax.legend(frameon=False, loc="upper right", labelcolor=ROXO)
f.text(0.5, 0.92, "Quantas equipes para fechar o ciclo?", ha="center", color=ROXO, fontweight="bold", fontsize=17)
f.savefig("res_sensib.png", transparent=True); plt.close(f)

# RPD médio por método
e = pd.read_csv(R + "experimento_rpd.csv")
m = e.groupby("algoritmo").rpd.mean()
ordem = [("LPT+NEH", 0.4978), ("IG", m["IG"]), ("AG", m["AG"]), ("ILS", m["ILS"]), ("VNS", m["VNS"])]
f = plt.figure(figsize=(7, 5), dpi=200); cartao(f)
ax = f.add_axes([0.17, 0.15, 0.76, 0.7]); ax.set_facecolor("none")
for i, (nome, v) in enumerate(ordem):
    cor = ROXO if nome == "VNS" else (CINZA if nome == "LPT+NEH" else LILAS)
    ax.barh(i, v, color=cor, height=0.62)
    ax.text(v + 0.01, i, f"{v:.2f}%".replace(".", ","), va="center", color=ROXO, fontweight="bold")
ax.set_yticks(range(5), [o[0] for o in ordem]); ax.set_xlim(0, 0.62)
ax.set_xlabel("RPD médio (%) · menor é melhor"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
f.text(0.5, 0.91, "18 instâncias × 30 execuções", ha="center", color=ROXO, fontweight="bold", fontsize=17)
f.savefig("res_rpd.png", transparent=True); plt.close(f)
print(ordem)
