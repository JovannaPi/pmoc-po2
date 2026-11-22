"""
Análise de sensibilidade: quantos dias o ciclo da Caso-base leva com 5 a 9
equipes. Usa o algoritmo de menor RPD médio no experimento final, com os
parâmetros calibrados (5 execuções por cenário), e compara com o plano do
sistema e com a execução real da regra atual. As vagas por prédio por dia
são mantidas (são um parâmetro do sistema atual).
"""
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSOS, TEMPO_EXPERIMENTO
from pmoc import Instancia, decodificar, regra_atual, dias_planejados, construtiva_lpt_neh
from metaheuristicas import ig, vns, ils, ag
from estilo import aplicar_estilo, COR, CINZA

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(BASE, "resultados")
EQUIPES = [5, 6, 7, 8, 9]
EXECUCOES = 5
FUNC = {"IG": ig, "VNS": vns, "ILS": ils, "AG": ag}


def carregar(m):
    inst = Instancia(os.path.join(BASE, "instancias", "Caso-base.txt"))
    inst.m = m
    return inst


def neh(m):
    construtiva_lpt_neh(carregar(m))  # calcula e guarda o cache para este m


def tarefa(args):
    m, alg, params, rep = args
    inst = carregar(m)
    s, _ = FUNC[alg](inst, tempo=TEMPO_EXPERIMENTO * inst.n, semente=100 + rep, **params)
    c = decodificar(s, inst.n, *inst.args())
    return dict(equipes=m, rep=rep, cmax=int(c[0]), dias=int(c[3]), deslocamento=int(c[2]))


def main():
    aplicar_estilo()
    ex = pd.read_csv(os.path.join(R, "experimento_rpd.csv"))
    alg = ex.groupby("algoritmo").rpd.mean().idxmin()
    with open(os.path.join(R, "parametros_finais.json")) as f:
        params = json.load(f)[alg]
    with Pool(PROCESSOS) as pool:
        pool.map(neh, EQUIPES)
        res = pool.map(tarefa, [(m, alg, params, r) for m in EQUIPES for r in range(EXECUCOES)])
    df = pd.DataFrame(res)
    linhas = []
    for m in EQUIPES:
        inst = carregar(m)
        c_ra = decodificar(regra_atual(inst), inst.n, *inst.args())
        d = df[df.equipes == m]
        linhas.append(dict(equipes=m, plano_sistema=dias_planejados(inst), regra_atual=int(c_ra[3]),
                           melhor_dias=int(d.dias.min()), media_dias=d.dias.mean(),
                           deslocamento_regra_atual=int(c_ra[2]), deslocamento_medio=d.deslocamento.mean()))
    tab = pd.DataFrame(linhas)
    tab["algoritmo"] = alg
    tab.to_csv(os.path.join(R, "sensibilidade_equipes.csv"), index=False)
    print(tab.to_string())

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(tab.equipes, tab.plano_sistema, "--o", color=CINZA, linewidth=2, label="Plano do sistema")
    ax.plot(tab.equipes, tab.regra_atual, "-o", color="#0b0b0b", linewidth=2, label="Regra atual (execução estimada)")
    ax.plot(tab.equipes, tab.melhor_dias, "-o", color=COR[alg], linewidth=2, label=f"{alg}")
    ax.set_xlabel("Número de equipes")
    ax.set_ylabel("Dias do ciclo")
    ax.set_xticks(EQUIPES)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(BASE, "figuras", "sensibilidade_equipes.pdf"))
    fig.savefig(os.path.join(BASE, "figuras", "sensibilidade_equipes.png"), dpi=200)


if __name__ == "__main__":
    main()
