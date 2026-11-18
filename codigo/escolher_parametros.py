"""
Consolida a calibração por grade (aula) e o refinamento com Optuna.
Ambas são avaliadas contra a MESMA referência: a melhor solução conhecida de
cada instância de calibração (mínimo entre todas as execuções das duas
etapas). Para cada algoritmo, escolhe-se a configuração de menor desvio
médio. Gera também o boxplot da calibração por grade.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from estilo import aplicar_estilo, COR, CINZA

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(BASE, "resultados")


def _v(x):
    return str(x).replace(".", ",")


def rotulo(alg, p):
    if alg == "IG":
        return f"{p['d']}\n{_v(p['tau'])}"
    if alg == "VNS":
        return f"{p['kmax']}\n{_v(p['fator_L'])}n"
    pct = _v("%g" % (p["intensidade"] * 100))
    return f"{pct}%\n{_v(p['fator_L'])}n"


EIXO_X = {"IG": "d  /  τ", "VNS": "kmax  /  L", "ILS": "intensidade  /  L"}


def main():
    aplicar_estilo()
    g = pd.read_csv(os.path.join(R, "calibracao_execucoes.csv"))
    o = pd.read_csv(os.path.join(R, "optuna_execucoes.csv"))
    ref = pd.concat([g[["instancia", "cmax"]], o[["instancia", "cmax"]]]).groupby("instancia").cmax.min()

    def desvios(df, chave):
        m = df.groupby([chave, "instancia"]).cmax.mean().reset_index()
        m["desvio"] = (m.cmax - m.instancia.map(ref)) / m.instancia.map(ref) * 100
        return m

    finais, tabela = {}, []
    fig, eixos = plt.subplots(1, 3, figsize=(7.5, 2.9), sharey=False, gridspec_kw={"width_ratios": [9, 6, 9]})
    for ax, alg in zip(eixos, ["IG", "VNS", "ILS"]):
        dg = desvios(g[g.algoritmo == alg], "parametros")
        rg = dg.groupby("parametros").desvio.mean().sort_values()
        do = desvios(o[o.algoritmo == alg], "parametros")
        ro = do.groupby("parametros").desvio.mean().sort_values()
        melhor_g = eval(rg.index[0])
        melhor_o = json.loads(ro.index[0])
        if ro.iloc[0] < rg.iloc[0]:
            escolha, origem = melhor_o, "Optuna"
        else:
            escolha, origem = melhor_g, "Grade"
        if alg == "IG":
            escolha["d"] = int(escolha["d"])
        if alg == "VNS":
            escolha["kmax"] = int(escolha["kmax"])
        finais[alg] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in escolha.items()}
        tabela.append(dict(algoritmo=alg, grade=json.dumps(melhor_g), desvio_grade=rg.iloc[0],
                           optuna=json.dumps({k: round(v, 3) for k, v in melhor_o.items()}),
                           desvio_optuna=ro.iloc[0], escolhido=origem))

        # boxplot (ordem da grade), melhor combinação destacada em vermelho
        ordem = sorted(rg.index, key=lambda s: list(eval(s).values()))
        dados = [dg[dg.parametros == p].desvio.values for p in ordem]
        bp = ax.boxplot(dados, patch_artist=True, widths=0.6,
                        medianprops=dict(color="#0b0b0b", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, markeredgecolor=CINZA))
        for caixa, p in zip(bp["boxes"], ordem):
            destaque = p == rg.index[0]
            caixa.set_facecolor("#e34948" if destaque else "#dcdcd6")
            caixa.set_edgecolor("#55554f")
        ax.set_xticks(range(1, len(ordem) + 1))
        ax.set_xticklabels([rotulo(alg, eval(p)) for p in ordem], fontsize=6.5)
        ax.set_xlabel(EIXO_X[alg], fontsize=8)
        ax.set_title(alg, fontsize=10)
        ax.set_ylabel("Desvio percentual relativo (%)" if alg == "IG" else "")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figuras", "calibracao_boxplot.pdf"))
    fig.savefig(os.path.join(BASE, "figuras", "calibracao_boxplot.png"), dpi=200)

    pd.DataFrame(tabela).to_csv(os.path.join(R, "calibracao_comparacao.csv"), index=False)
    with open(os.path.join(R, "parametros_finais.json"), "w") as f:
        json.dump(finais, f, indent=2)
    print(pd.DataFrame(tabela).to_string())
    print(json.dumps(finais, indent=2))


if __name__ == "__main__":
    main()
