"""
Refinamento da calibração com Optuna (amostrador TPE).

Para cada meta-heurística, o Optuna propõe valores de parâmetros num
intervalo contínuo/inteiro (mais amplo que a grade). Cada tentativa executa
o algoritmo 2 vezes em cada instância de calibração; o valor minimizado é
o desvio percentual médio em relação à melhor solução conhecida da
instância (obtida na calibração por grade).
"""
import json
import os
import sys
from multiprocessing import Pool

import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSOS
from calibracao import BASE, TEMPO_POR_APARELHO, instancias_calibracao, FUNC
from pmoc import Instancia, decodificar

TENTATIVAS = 12
REPS = 2


def rodar(args):
    caminho, alg, params, rep = args
    inst = Instancia(caminho)
    s, _ = FUNC[alg](inst, tempo=TEMPO_POR_APARELHO * inst.n, semente=5000 + rep, **params)
    return os.path.basename(caminho)[:-4], int(decodificar(s, inst.n, *inst.args())[0])


def espaco(alg, trial):
    if alg == "IG":
        return {"d": trial.suggest_int("d", 2, 10), "tau": trial.suggest_float("tau", 0.05, 1.5)}
    if alg == "VNS":
        return {"kmax": trial.suggest_int("kmax", 2, 5), "fator_L": trial.suggest_float("fator_L", 0.25, 3.0)}
    return {"intensidade": trial.suggest_float("intensidade", 0.005, 0.10),
            "fator_L": trial.suggest_float("fator_L", 0.25, 3.0)}


def main():
    calib, _ = instancias_calibracao()
    grade = pd.read_csv(os.path.join(BASE, "resultados", "calibracao_execucoes.csv"))
    melhor = grade.groupby("instancia").cmax.min().to_dict()
    pool = Pool(PROCESSOS)
    registros, melhores, execucoes = [], {}, []
    for alg in ["IG", "VNS", "ILS"]:
        def objetivo(trial):
            params = espaco(alg, trial)
            tarefas = [(c, alg, params, r) for c in calib for r in range(REPS)]
            res = pool.map(rodar, tarefas, chunksize=1)
            for nome, cmax in res:
                melhor[nome] = min(melhor[nome], cmax)
                execucoes.append(dict(algoritmo=alg, tentativa=trial.number,
                                      parametros=json.dumps(params), instancia=nome, cmax=cmax))
            df = pd.DataFrame(res, columns=["instancia", "cmax"])
            g = df.groupby("instancia").cmax.mean()
            desvio = float(((g - g.index.map(melhor)) / g.index.map(melhor) * 100).mean())
            registros.append(dict(algoritmo=alg, tentativa=trial.number, desvio=desvio, **params))
            print(alg, trial.number, params, round(desvio, 3), flush=True)
            return desvio

        estudo = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=7))
        estudo.optimize(objetivo, n_trials=TENTATIVAS)
        melhores[alg] = {"params": estudo.best_params, "desvio": estudo.best_value}
    pool.close()
    pd.DataFrame(execucoes).to_csv(os.path.join(BASE, "resultados", "optuna_execucoes.csv"), index=False)
    pd.DataFrame(registros).to_csv(os.path.join(BASE, "resultados", "optuna_tentativas.csv"), index=False)
    with open(os.path.join(BASE, "resultados", "optuna_melhores.json"), "w") as f:
        json.dump(melhores, f, indent=2)
    print(json.dumps(melhores, indent=2))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
