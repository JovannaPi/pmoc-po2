"""
Calibração dos parâmetros (metodologia da aula de Calibração):
  1. sorteia 30% das instâncias geradas;
  2. testa todas as combinações de parâmetros de cada meta-heurística;
  3. executa 5 vezes cada combinação em cada instância;
  4. desvio = (média - melhor da instância) / melhor da instância * 100;
  5. escolhe a combinação com menor desvio médio.
"""
import glob
import itertools
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pmoc import Instancia, decodificar
from metaheuristicas import ig, vns, ils

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
from config import TEMPO_POR_APARELHO, PROCESSOS
EXECUCOES = 5

GRADES = {
    "IG": {"d": [2, 4, 6], "tau": [0.1, 0.4, 0.8]},
    "VNS": {"kmax": [3, 5], "fator_L": [0.5, 1.0, 2.0]},
    "ILS": {"intensidade": [0.01, 0.025, 0.05], "fator_L": [0.5, 1.0, 2.0]},
}
FUNC = {"IG": ig, "VNS": vns, "ILS": ils}


def instancias_calibracao():
    todas = sorted(glob.glob(os.path.join(BASE, "instancias", "PMOC-*.txt")))
    random.seed(42)
    k = round(0.3 * len(todas))
    return sorted(random.sample(todas, k)), todas


def tarefa(args):
    caminho, alg, params, rep = args
    inst = Instancia(caminho)
    s, _ = FUNC[alg](inst, tempo=TEMPO_POR_APARELHO * inst.n, semente=1000 + rep, **params)
    cmax, _, desl, dias = decodificar(s, inst.n, *inst.args())
    return dict(instancia=os.path.basename(caminho)[:-4], algoritmo=alg,
                parametros=str(params), rep=rep, cmax=int(cmax), dias=int(dias),
                deslocamento=int(desl), **params)


def main():
    calib, _ = instancias_calibracao()
    print("Instâncias de calibração:", [os.path.basename(c) for c in calib])
    tarefas = []
    for alg, grade in GRADES.items():
        chaves = list(grade)
        for valores in itertools.product(*grade.values()):
            params = dict(zip(chaves, valores))
            for c in calib:
                for rep in range(EXECUCOES):
                    tarefas.append((c, alg, params, rep))
    # Retomada: execuções já salvas não são repetidas
    arq = os.path.join(BASE, "resultados", "calibracao_execucoes.csv")
    res = pd.read_csv(arq).to_dict("records") if os.path.exists(arq) else []
    feitas = {(r["instancia"], r["algoritmo"], r["parametros"], r["rep"]) for r in res}
    tarefas = [t for t in tarefas
               if (os.path.basename(t[0])[:-4], t[1], str(t[2]), t[3]) not in feitas]
    print(len(tarefas), "execuções pendentes", flush=True)
    with Pool(PROCESSOS) as pool:
        for i, r in enumerate(pool.imap_unordered(tarefa, tarefas, chunksize=1)):
            res.append(r)
            if i % 25 == 0:
                print(i, flush=True)
                pd.DataFrame(res).to_csv(arq, index=False)
    df = pd.DataFrame(res)
    df.to_csv(arq, index=False)

    # Desvio percentual relativo por (instância, combinação), por algoritmo
    linhas = []
    for alg in GRADES:
        d = df[df.algoritmo == alg]
        melhor = d.groupby("instancia").cmax.min()
        g = d.groupby(["parametros", "instancia"]).cmax.mean().reset_index()
        g["melhor"] = g.instancia.map(melhor)
        g["desvio"] = (g.cmax - g.melhor) / g.melhor * 100
        g["algoritmo"] = alg
        linhas.append(g)
    res = pd.concat(linhas)
    res.to_csv(os.path.join(BASE, "resultados", "calibracao_desvios.csv"), index=False)
    resumo = res.groupby(["algoritmo", "parametros"]).desvio.mean().reset_index()
    resumo = resumo.sort_values(["algoritmo", "desvio"])
    resumo.to_csv(os.path.join(BASE, "resultados", "calibracao_resumo.csv"), index=False)
    print(resumo.to_string())


if __name__ == "__main__":
    main()
