"""
Experimento final: IG, VNS, ILS e AG calibrados, 30 execuções independentes em
cada instância de teste (todas as que não foram usadas na calibração),
mais as heurísticas determinísticas (regra atual do sistema e LPT+NEH).
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSOS
from calibracao import BASE, TEMPO_POR_APARELHO, instancias_calibracao, FUNC
from pmoc import (Instancia, decodificar, regra_atual, construtiva_lpt_neh,
                  dias_planejados, limite_inferior)

EXECUCOES = 30
INST_CONVERGENCIA = []  # definida em main(): ALECE-base + maior instância de teste


def tarefa(args):
    caminho, alg, params, rep, registrar_conv = args
    inst = Instancia(caminho)
    nome = os.path.basename(caminho)[:-4]
    s, _, hist = FUNC[alg](inst, tempo=TEMPO_POR_APARELHO * inst.n, semente=rep,
                           historico=True, **params)
    cmax, _, desl, dias = decodificar(s, inst.n, *inst.args())
    linha = dict(instancia=nome, n=inst.n, m=inst.m, algoritmo=alg, rep=rep,
                 cmax=int(cmax), dias=int(dias), deslocamento=int(desl))
    conv = [dict(instancia=nome, algoritmo=alg, rep=rep, tempo=t, f=int(f))
            for t, f in hist] if registrar_conv else []
    if nome == "ALECE-base":  # guarda a solução para as figuras do estudo de caso
        pasta = os.path.join(BASE, "resultados", "solucoes")
        os.makedirs(pasta, exist_ok=True)
        np.save(os.path.join(pasta, f"{nome}_{alg}_{rep}.npy"), s)
    return linha, conv


def main():
    calib, todas = instancias_calibracao()
    teste = [c for c in todas if c not in calib]
    teste.append(os.path.join(BASE, "instancias", "ALECE-base.txt"))
    maior = max((c for c in teste if "-m7-" in c), key=lambda c: (Instancia(c).n, c))
    INST_CONVERGENCIA[:] = ["ALECE-base", os.path.basename(maior)[:-4]]
    print("Convergência registrada em:", INST_CONVERGENCIA, flush=True)
    with open(os.path.join(BASE, "resultados", "parametros_finais.json")) as f:
        PARAMS = json.load(f)

    # Heurísticas determinísticas
    det = []
    print("Heurísticas determinísticas...", flush=True)
    for c in teste:
        inst = Instancia(c)
        nome = os.path.basename(c)[:-4]
        for alg, s in [("Regra atual", regra_atual(inst)), ("LPT+NEH", construtiva_lpt_neh(inst))]:
            cmax, _, desl, dias = decodificar(s, inst.n, *inst.args())
            det.append(dict(instancia=nome, n=inst.n, m=inst.m, algoritmo=alg,
                            cmax=int(cmax), dias=int(dias), deslocamento=int(desl),
                            dias_planejados=dias_planejados(inst), lb=limite_inferior(inst)))
    pd.DataFrame(det).to_csv(os.path.join(BASE, "resultados", "deterministicos.csv"), index=False)

    tarefas = [(c, alg, PARAMS[alg], r, os.path.basename(c)[:-4] in INST_CONVERGENCIA)
               for c in teste for alg in ["IG", "VNS", "ILS", "AG"] for r in range(EXECUCOES)]
    # Retomada: execuções já salvas não são repetidas
    arq = os.path.join(BASE, "resultados", "experimento_execucoes.csv")
    arq_c = os.path.join(BASE, "resultados", "convergencia.csv")
    linhas = pd.read_csv(arq).to_dict("records") if os.path.exists(arq) else []
    conv = pd.read_csv(arq_c).to_dict("records") if os.path.exists(arq_c) else []
    feitas = {(l["instancia"], l["algoritmo"], l["rep"]) for l in linhas}
    tarefas = [t for t in tarefas if (os.path.basename(t[0])[:-4], t[1], t[3]) not in feitas]
    print(len(tarefas), "execuções pendentes", flush=True)
    with Pool(PROCESSOS) as pool:
        for i, (l, cv) in enumerate(pool.imap_unordered(tarefa, tarefas, chunksize=1)):
            linhas.append(l)
            conv.extend(cv)
            if i % 20 == 0:
                print(i, flush=True)
                pd.DataFrame(linhas).to_csv(arq, index=False)
                pd.DataFrame(conv).to_csv(arq_c, index=False)
    pd.DataFrame(linhas).to_csv(arq, index=False)
    pd.DataFrame(conv).to_csv(arq_c, index=False)
    print("fim")


if __name__ == "__main__":
    main()
