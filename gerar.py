import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
import segno

ROXO = "#522276"; MEDIO = "#885da8"; LILAS = "#a88ae1"; CLARO = "#e2d1ed"; BRANCO = "#ffffff"
GELO = "#9fd8ee"; CINZA = "#cfc6d8"
plt.rcParams["font.family"] = "DejaVu Sans"


def caixa(ax, x, y, w, h, cor, r=0.05, ec="none", lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=cor, ec=ec, lw=lw))


def ar_frio(ax, x0, x1, y, n=3, dy=0.09, cor=GELO, lw=5, amp=0.025):
    xs = np.linspace(x0, x1, 120)
    for i in range(n):
        ax.plot(xs, y - i * dy + amp * np.sin((xs - x0) / (x1 - x0) * 4 * np.pi),
                color=cor, lw=lw, solid_capstyle="round")


def hiwall(ax, x, y, w, h, corpo=BRANCO, borda=CINZA):
    caixa(ax, x, y, w, h, corpo, r=h * 0.35, ec=borda, lw=3)
    ax.plot([x + w * 0.08, x + w * 0.92], [y + h * 0.28] * 2, color=borda, lw=4, solid_capstyle="round")
    ax.add_patch(Circle((x + w * 0.88, y + h * 0.66), h * 0.07, color="#6fd18b"))


def pisoteto(ax, x, y, w, h, corpo=BRANCO, borda=CINZA):
    ax.add_patch(Rectangle((x - w * 0.1, y + h), w * 1.2, h * 0.12, color=borda))  # teto
    caixa(ax, x, y, w, h, corpo, r=h * 0.2, ec=borda, lw=3)
    for k in range(5):
        ax.plot([x + w * (0.12 + 0.17 * k), x + w * (0.2 + 0.17 * k)], [y + h * 0.25] * 2,
                color=borda, lw=4, solid_capstyle="round")


def cassete(ax, cx, cy, s, corpo=BRANCO, borda=CINZA):
    caixa(ax, cx - s / 2, cy - s / 2, s, s, corpo, r=s * 0.08, ec=borda, lw=3)
    caixa(ax, cx - s * 0.22, cy - s * 0.22, s * 0.44, s * 0.44, CLARO, r=s * 0.04, ec=borda, lw=2)
    g = s * 0.36
    for (dx, dy, horiz) in [(0, g, 1), (0, -g, 1), (g, 0, 0), (-g, 0, 0)]:
        if horiz:
            ax.plot([cx - s * 0.3, cx + s * 0.3], [cy + dy] * 2, color=borda, lw=6, solid_capstyle="round")
        else:
            ax.plot([cx + dx] * 2, [cy - s * 0.3, cy + s * 0.3], color=borda, lw=6, solid_capstyle="round")


def fig(w, h):
    f = plt.figure(figsize=(w, h), dpi=200)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return f, ax


def salvar(f, nome):
    f.savefig(nome, transparent=True); plt.close(f)


# 1) Tipos de aparelho e tempos (slide 3)
f, ax = fig(5, 5)
cards = [("Hi-wall pequeno", "9–12 mil BTU", "2h", "hi"),
         ("Hi-wall grande", "18–24 mil BTU", "2h30", "hiG"),
         ("Piso-teto", "", "3h", "pt"),
         ("Cassete", "", "3h30", "ca")]
pos = [(0.1, 2.6), (2.6, 2.6), (0.1, 0.1), (2.6, 0.1)]
for (nome, sub, t, tipo), (x, y) in zip(cards, pos):
    caixa(ax, x, y, 2.3, 2.3, CLARO, r=0.18)
    if tipo == "hi":
        hiwall(ax, x + 0.45, y + 1.35, 1.4, 0.5)
    elif tipo == "hiG":
        hiwall(ax, x + 0.2, y + 1.3, 1.9, 0.62)
    elif tipo == "pt":
        pisoteto(ax, x + 0.2, y + 1.35, 1.9, 0.48)
    else:
        cassete(ax, x + 1.15, y + 1.6, 0.9)
    ax.text(x + 1.15, y + 0.95, nome, ha="center", va="center", fontsize=15, color=ROXO, fontweight="bold")
    if sub:
        ax.text(x + 1.15, y + 0.68, sub, ha="center", va="center", fontsize=11, color=MEDIO)
    caixa(ax, x + 0.65, y + 0.12, 1.0, 0.4, ROXO, r=0.2)
    ax.text(x + 1.15, y + 0.32, t, ha="center", va="center", fontsize=17, color=BRANCO, fontweight="bold")
salvar(f, "tipos_ac.png")

# 2) Plano x real (slide 2)
f, ax = fig(5, 5)
caixa(ax, 0.1, 0.1, 4.8, 4.8, CLARO, r=0.3)
hiwall(ax, 1.3, 3.75, 2.4, 0.75)
ar_frio(ax, 1.6, 3.4, 3.5, n=2, dy=0.15, lw=4)
ax.text(2.5, 2.75, "Duração do ciclo de preventiva", ha="center", fontsize=14, color=ROXO, fontweight="bold")
for i, (rot, d, cor) in enumerate([("Plano do sistema", 45, LILAS), ("Execução real", 74, ROXO)]):
    y = 1.85 - i * 1.05
    ax.text(0.45, y + 0.42, rot, fontsize=13, color=ROXO)
    caixa(ax, 0.45, y - 0.12, 3.9 * d / 74, 0.45, cor, r=0.12)
    ax.text(0.45 + 3.9 * d / 74 - 0.12, y + 0.1, f"{d} dias", ha="right", va="center",
            fontsize=15, color=BRANCO, fontweight="bold")
salvar(f, "plano_real.png")

# 3) Mapa dos prédios (slide 4)
f, ax = fig(5, 6)
caixa(ax, 0.1, 0.1, 4.8, 5.8, CLARO, r=0.3)
ax.text(2.5, 5.45, "5 prédios · 1050 aparelhos", ha="center", fontsize=15, color=ROXO, fontweight="bold")
ax.plot([0.25, 4.75], [2.35, 2.75], color=BRANCO, lw=16, solid_capstyle="round")
ax.text(2.5, 2.25, "Av. Pontes Vieira", rotation=5, fontsize=10, color=MEDIO, style="italic", ha="center")
pred = [("SEDE", 327, 8, 0.95, 4.25, 0.55), ("AI", 157, 4, 2.5, 4.25, 0.45), ("AII", 222, 5, 4.05, 4.25, 0.5),
        ("AIII", 173, 4, 1.55, 1.25, 0.48), ("AIV", 171, 4, 3.45, 1.25, 0.48)]
for nome, n, v, x, y, r in pred:
    ax.add_patch(Circle((x, y), r, color=ROXO if nome == "SEDE" else MEDIO))
    ax.text(x, y, nome, ha="center", va="center", fontsize=14, color=BRANCO, fontweight="bold")
    ax.text(x, y - r - 0.2, f"{n} aparelhos", fontsize=10, color=ROXO, ha="center", va="center")
    ax.text(x, y - r - 0.42, f"{v} vagas/dia", fontsize=10, color=MEDIO, ha="center", va="center")
salvar(f, "mapa_predios.png")

# 4) Matriz de deslocamento (slide 7)
D = np.array([[8, 8, 10, 15, 20], [8, 8, 10, 12, 18], [10, 10, 8, 15, 18], [15, 12, 15, 8, 15], [20, 18, 18, 15, 8]])
nomes = ["SEDE", "AI", "AII", "AIII", "AIV"]
f, ax = fig(6, 5.6)
ax.text(3.0, 5.3, "Deslocamento entre prédios (min)", ha="center", fontsize=15, color=BRANCO, fontweight="bold")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("r", [CLARO, LILAS, MEDIO, "#2e0f47"])
c = 0.9; x0 = 1.0; y0 = 4.6
for j, nm in enumerate(nomes):
    ax.text(x0 + c * j + c / 2, y0 + 0.25, nm, ha="center", fontsize=12, color=BRANCO, fontweight="bold")
    ax.text(x0 - 0.12, y0 - c * j - c / 2, nm, ha="right", va="center", fontsize=12, color=BRANCO, fontweight="bold")
    for k in range(5):
        v = D[j, k]; t = (v - 8) / 12
        caixa(ax, x0 + c * k + 0.04, y0 - c * (j + 1) + 0.04, c - 0.08, c - 0.08, cmap(0.15 + 0.85 * t), r=0.1)
        ax.text(x0 + c * k + c / 2, y0 - c * j - c / 2, str(v), ha="center", va="center", fontsize=15,
                color=ROXO if t < 0.45 else BRANCO, fontweight="bold")
salvar(f, "matriz_desloc.png")

# 5) Ilustração para a conclusão (círculo)
f, ax = fig(5, 5)
ax.add_patch(Circle((2.5, 2.5), 2.45, color=CLARO))
hiwall(ax, 0.9, 2.9, 3.2, 1.0)
ar_frio(ax, 1.2, 3.8, 2.55, n=3, dy=0.28, lw=7, amp=0.06)
ax.add_patch(Circle((3.7, 1.15), 0.55, color=ROXO))
ax.plot([3.45, 3.65, 3.97], [1.17, 0.97, 1.37], color=BRANCO, lw=9, solid_capstyle="round", solid_joinstyle="round")
salvar(f, "ac_check.png")

# 6) Ilustração para o encerramento (cassete no teto)
f, ax = fig(5, 6)
caixa(ax, 0.1, 0.1, 4.8, 5.8, CLARO, r=0.3)
ax.add_patch(Rectangle((0.1, 4.6), 4.8, 0.18, color=CINZA))
cassete(ax, 2.5, 3.9, 1.6)
ar_frio(ax, 0.8, 4.2, 2.65, n=3, dy=0.32, lw=7, amp=0.07)
for k, (x, y) in enumerate([(1.1, 1.2), (2.5, 0.9), (3.9, 1.2)]):
    ax.text(x, y, "❄", ha="center", va="center", fontsize=34, color=MEDIO)
salvar(f, "ac_cassete.png")

# 7) QR code do repositório
segno.make("https://github.com/JovannaPi/pmoc-po2", error="m").save(
    "qr_github.png", scale=20, border=2, dark=ROXO, light=BRANCO)
print("ok")
