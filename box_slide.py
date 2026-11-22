import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
R="/home/claude/pmoc-po2/resultados/"
ROXO="#522276"; LILAS="#a88ae1"; CLARO="#e2d1ed"; CINZA="#9a8fa6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.edgecolor":CINZA,"axes.labelcolor":ROXO,
  "xtick.color":ROXO,"ytick.color":ROXO,"axes.spines.top":False,"axes.spines.right":False})
ex=pd.read_csv(R+"experimento_execucoes.csv"); det=pd.read_csv(R+"deterministicos.csv")
neh=det[det.algoritmo=="LPT+NEH"].set_index("instancia").cmax
ex["m"]=100*(ex.instancia.map(neh)-ex.cmax)/ex.instancia.map(neh)
A=["IG","VNS","ILS","AG"]
f=plt.figure(figsize=(7,5),dpi=200)
f.patches.append(FancyBboxPatch((0.005,0.005),0.99,0.99,boxstyle="round,pad=0,rounding_size=0.035",transform=f.transFigure,fc=CLARO,ec="none",zorder=-10))
ax=f.add_axes([0.15,0.13,0.8,0.7]); ax.set_facecolor("none")
d=[ex[ex.algoritmo==a].m.values for a in A]
bp=ax.boxplot(d,widths=0.55,patch_artist=True,medianprops=dict(color="white",lw=2.5),
  flierprops=dict(marker="o",ms=5,alpha=0.55,mec="none"),whiskerprops=dict(color=ROXO,lw=1.5),capprops=dict(color=ROXO,lw=1.5))
for b,fl,a in zip(bp["boxes"],bp["fliers"],A):
    c=ROXO if a=="VNS" else LILAS; b.set_facecolor(c); b.set_edgecolor(ROXO); fl.set_markerfacecolor(c)
for i,x in enumerate(d,1): ax.plot(i,x.mean(),"D",ms=8,color="white",mec=ROXO,mew=1.5,zorder=5)
ax.set_xticks(range(1,5),A); ax.set_ylabel("Melhoria sobre a LPT+NEH (%)")
ax.axhline(0,color=CINZA,lw=1); ax.grid(axis="y",color="white",lw=1.5); ax.set_axisbelow(True)
f.text(0.5,0.9,"Boxplot: 18 instâncias × 30 execuções",ha="center",color=ROXO,fontweight="bold",fontsize=17)
f.savefig("res_box.png",transparent=True)
