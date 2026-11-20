"""
Modelo de Programação Linear Inteira Mista (MILP) do problema PMOC e
validação das meta-heurísticas em instâncias pequenas.

Formulação (resolvida com Python-MIP + CBC):

  Conjuntos: J aparelhos, K equipes, D dias (D = nº de dias de uma solução
  viável conhecida, a da regra atual), B prédios.
  Variáveis:
    x[j,k,d] = 1 se o aparelho j é atendido pela equipe k no dia d
    z[j,k,d] = 1 se j é o primeiro aparelho da equipe k no dia d
    y[i,j,k,d] = 1 se j é atendido logo após i pela equipe k no dia d
    S[j] >= 0 : início do atendimento de j (minutos desde o início do dia)
    Cmax >= 0 : makespan em minutos úteis acumulados
  min  Cmax
  s.a. (1) sum_{k,d} x[j,k,d] = 1                                   para todo j
       (2) z[j,k,d] + sum_{i!=j} y[i,j,k,d] = x[j,k,d]               para todo j,k,d
       (3) sum_{l!=j} y[j,l,k,d] <= x[j,k,d]                          para todo j,k,d
       (4) sum_j z[j,k,d] <= 1                                        para todo k,d
       (5) S[j] >= s[SEDE, b_j]                                       para todo j
       (6) S[j] >= S[i] + p_i + s[b_i,b_j] - M (1 - sum_{k,d} y[i,j,k,d])   i != j
       (7) S[j] + p_j <= H                                            para todo j
       (8) sum_{j: b_j = b} sum_k x[j,k,d] <= c_b                     para todo b,d
       (9) Cmax >= sum_{k,d} d H x[j,k,d] + S[j] + p_j                para todo j
      (10) sum_j x[j,k,d] >= sum_j x[j,k+1,d]   (quebra de simetria entre equipes)
      (11) sum_j (p_j + smin_j) x[j,k,d] <= H                        para todo k,d
      (12) Cmax >= d H sum_j z[j,k,d] + sum_j (p_j + smin_j) x[j,k,d]  para todo k,d
  em que smin_j é o menor deslocamento possível até o aparelho j; (6) usa
  M_ij = H + s[b_i,b_j] - s[SEDE,b_j]. (11) e (12) são desigualdades válidas.

O MILP admite qualquer programação viável; as meta-heurísticas exploram as
programações geradas pelo decodificador. Se elas atingem o ótimo do MILP,
fica validado que o decodificador não exclui as melhores soluções nessas
instâncias e que as meta-heurísticas as encontram.
"""
import os
import sys
import time
from multiprocessing import Pool

import mip
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmoc import Instancia, decodificar, regra_atual, construtiva_lpt_neh
from metaheuristicas import ig, vns, ils, ag
from gerar_instancias import gerar

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PASTA = os.path.join(BASE, "instancias_pequenas")
LIMITE_MILP = 600      # s por instância
TEMPO_MH = 1.0         # s por execução das meta-heurísticas
EXECUCOES = 10


def resolver_milp(inst, D, limite=LIMITE_MILP):
    n, m, H = inst.n, inst.m, inst.H
    b, p, T, cap = inst.predio, inst.p, inst.T, inst.cap
    J, K, Ds = range(n), range(m), range(D)
    M = H + int(T.max())
    mod = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    mod.verbose = 0
    mod.threads = 1
    x = {(j, k, d): mod.add_var(var_type=mip.BINARY) for j in J for k in K for d in Ds}
    z = {(j, k, d): mod.add_var(var_type=mip.BINARY) for j in J for k in K for d in Ds}
    y = {(i, j, k, d): mod.add_var(var_type=mip.BINARY)
         for i in J for j in J if i != j for k in K for d in Ds}
    S = [mod.add_var(lb=0) for _ in J]
    Cmax = mod.add_var(lb=0)
    mod.objective = Cmax
    for j in J:
        mod += mip.xsum(x[j, k, d] for k in K for d in Ds) == 1
        mod += S[j] >= int(T[0, b[j]])
        mod += S[j] + int(p[j]) <= H
        mod += Cmax >= mip.xsum(d * H * x[j, k, d] for k in K for d in Ds) + S[j] + int(p[j])
        for k in K:
            for d in Ds:
                mod += z[j, k, d] + mip.xsum(y[i, j, k, d] for i in J if i != j) == x[j, k, d]
                mod += mip.xsum(y[j, l, k, d] for l in J if l != j) <= x[j, k, d]
    for k in K:
        for d in Ds:
            mod += mip.xsum(z[j, k, d] for j in J) <= 1
    for i in J:
        for j in J:
            if i != j:
                Mij = H + int(T[b[i], b[j]]) - int(T[0, b[j]])
                mod += S[j] >= S[i] + int(p[i]) + int(T[b[i], b[j]]) - Mij * (
                    1 - mip.xsum(y[i, j, k, d] for k in K for d in Ds))
    # Desigualdades válidas (fortalecem a relaxação linear): a carga de cada
    # equipe em cada dia, contando o menor deslocamento possível até cada
    # aparelho, cabe na jornada e limita o Cmax por baixo.
    smin = [int(min(T[a, b[j]] for a in range(T.shape[0]))) for j in J]
    for k in K:
        for d in Ds:
            carga = mip.xsum((int(p[j]) + smin[j]) * x[j, k, d] for j in J)
            mod += carga <= H
            mod += Cmax >= d * H * mip.xsum(z[j, k, d] for j in J) + carga
    for bb in range(T.shape[0]):
        for d in Ds:
            mod += mip.xsum(x[j, k, d] for j in J if b[j] == bb for k in K) <= int(cap[bb])
    for k in range(m - 1):
        for d in Ds:
            mod += mip.xsum(x[j, k, d] for j in J) >= mip.xsum(x[j, k + 1, d] for j in J)
    t = time.time()
    st = mod.optimize(max_seconds=limite)
    return dict(status=st.name, cmax_milp=mod.objective_value, lb_milp=mod.objective_bound,
                tempo_milp=time.time() - t)


def tarefa(caminho):
    inst = Instancia(caminho)
    nome = os.path.basename(caminho)[:-4]
    ra = regra_atual(inst)
    c_ra = decodificar(ra, inst.n, *inst.args())
    D = int(c_ra[3])  # nº de dias de uma solução viável (limite superior)
    linha = dict(instancia=nome, n=inst.n, m=inst.m, dias_ub=D, cmax_regra_atual=int(c_ra[0]))
    linha.update(resolver_milp(inst, D))
    linha["cmax_neh"] = int(decodificar(construtiva_lpt_neh(inst), inst.n, *inst.args())[0])
    for nome_alg, f, kw in [("IG", ig, {}), ("VNS", vns, {}), ("ILS", ils, {}), ("AG", ag, {})]:
        vals = []
        for r in range(EXECUCOES):
            s, _ = f(inst, tempo=TEMPO_MH, semente=r, **PARAMS[nome_alg])
            vals.append(int(decodificar(s, inst.n, *inst.args())[0]))
        linha[f"{nome_alg}_melhor"] = min(vals)
        linha[f"{nome_alg}_media"] = float(np.mean(vals))
        linha[f"{nome_alg}_acertos"] = sum(v <= round(linha["cmax_milp"]) for v in vals)
    print(nome, linha["status"], linha["cmax_milp"], {k: v for k, v in linha.items() if k.endswith("melhor")}, flush=True)
    return linha


PARAMS = {}


def main():
    import json
    global PARAMS
    arq_p = os.path.join(BASE, "resultados", "parametros_finais.json")
    if os.path.exists(arq_p):
        with open(arq_p) as f:
            PARAMS = json.load(f)
    for a, padrao in [("IG", {"d": 2, "tau": 0.1}), ("VNS", {"kmax": 3, "fator_L": 2.0}),
                      ("ILS", {"intensidade": 0.025, "fator_L": 1.0}), ("AG", {"tam_pop": 50, "pm": 0.2})]:
        PARAMS.setdefault(a, padrao)
    os.makedirs(PASTA, exist_ok=True)
    semente = 500
    for n in [6, 8, 10, 12]:
        for m in [1, 2]:
            for r in (1, 2):
                gerar(n, m, semente, f"PEQ-n{n}-m{m}-{r}", PASTA)
                semente += 1
    insts = sorted(os.path.join(PASTA, f) for f in os.listdir(PASTA) if f.endswith(".txt"))
    from config import PROCESSOS
    with Pool(PROCESSOS) as pool:
        linhas = pool.map(tarefa, insts, chunksize=1)
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(BASE, "resultados", "validacao_exata.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
