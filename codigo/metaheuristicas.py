"""
Meta-heurísticas para o problema PMOC (representação por permutação):
  - IG  (Iterated Greedy, Ruiz & Stützle, 2007)
  - VNS (Variable Neighborhood Search, Mladenović & Hansen, 1997)
  - ILS (Iterated Local Search, Lourenço et al., 2003)

Todas partem da mesma solução inicial (LPT + NEH) e usam o mesmo critério
de parada: tempo de CPU proporcional ao número de aparelhos.
"""
import time
import numpy as np
from numba import njit
from pmoc import (avaliar, avaliar_limite, melhor_insercao_rapida, inserir,
                  construtiva_lpt_neh)


# ---------------------------------------------------------------- IG
@njit(cache=True)
def destruir_construir(seq, d, predio, p, T, cap, m, H, maxD):
    n = len(seq)
    rem = np.random.choice(n, d, replace=False)
    marca = np.zeros(n, np.bool_)
    for i in rem:
        marca[i] = True
    removidos = np.empty(d, np.int64)
    parcial = np.empty(n, np.int64)
    L = 0
    r = 0
    for i in range(n):
        if marca[i]:
            removidos[r] = seq[i]
            r += 1
        else:
            parcial[L] = seq[i]
            L += 1
    np.random.shuffle(removidos)  # ordem de remoção aleatória
    buf = np.empty(n, np.int64)
    f = 0.0
    for j in removidos:
        pos, f = melhor_insercao_rapida(parcial, L, j, predio, p, T, cap, m, H, maxD)
        inserir(parcial, L, j, pos)
        L += 1
    return parcial, f


def ig(inst, d=4, tau=0.4, tempo=5.0, semente=0, historico=False):
    np.random.seed(semente)
    _seed_numba(semente)
    args = inst.args()
    s = construtiva_lpt_neh(inst)
    fs = avaliar(s, inst.n, *args)
    best, fbest = s.copy(), fs
    Temp = tau * inst.p.sum() / (inst.n * 10.0)  # temperatura (adaptada)
    hist = [(0.0, fbest)]
    t0 = time.process_time()
    while time.process_time() - t0 < tempo:
        s2, f2 = destruir_construir(s, d, *args)
        if f2 < fs or np.random.rand() < np.exp(-(f2 - fs) / Temp):
            s, fs = s2, f2
            if fs < fbest:
                best, fbest = s.copy(), fs
                hist.append((time.process_time() - t0, fbest))
    return (best, fbest, hist) if historico else (best, fbest)


# ---------------------------------------------------- busca local (VND)
@njit(cache=True)
def busca_local(seq, f, Lmax, predio, p, T, cap, m, H, maxD):
    """Descida com amostragem aleatória nas vizinhanças de troca (swap) e
    inserção; para após Lmax tentativas consecutivas sem melhoria."""
    n = len(seq)
    s = seq.copy()
    falhas = 0
    viz = 0  # 0 = troca, 1 = inserção (VND: muda de vizinhança ao falhar)
    while falhas < Lmax:
        a = np.random.randint(n)
        b = np.random.randint(n)
        if a == b:
            continue
        s2 = s.copy()
        if viz == 0:
            s2[a], s2[b] = s2[b], s2[a]
        else:
            j = s2[a]
            if a < b:
                for i in range(a, b):
                    s2[i] = s2[i + 1]
            else:
                for i in range(a, b, -1):
                    s2[i] = s2[i - 1]
            s2[b] = j
        f2 = avaliar_limite(s2, n, int(f), predio, p, T, cap, m, H, maxD)
        if f2 < f:
            s, f = s2, f2
            falhas = 0
            viz = 0
        else:
            falhas += 1
            if falhas % 2 == 0:
                viz = 1 - viz
    return s, f


# ---------------------------------------------------------------- VNS
@njit(cache=True)
def shaking(seq, k):
    """Cinco vizinhanças, da mais local para a mais global (aula de VNS)."""
    n = len(seq)
    s = seq.copy()
    if k == 1:  # troca de dois aparelhos
        a, b = np.random.choice(n, 2, replace=False)
        s[a], s[b] = s[b], s[a]
    elif k == 2:  # inserção de um aparelho no início
        a = np.random.randint(1, n)
        j = s[a]
        for i in range(a, 0, -1):
            s[i] = s[i - 1]
        s[0] = j
    elif k == 3:  # bloco de 4 movido para o início
        a = np.random.randint(1, n - 3)
        bloco = s[a:a + 4].copy()
        resto = np.concatenate((s[:a], s[a + 4:]))
        s = np.concatenate((bloco, resto))
    elif k == 4:  # bloco de 4 movido para o final
        a = np.random.randint(0, n - 4)
        bloco = s[a:a + 4].copy()
        resto = np.concatenate((s[:a], s[a + 4:]))
        s = np.concatenate((resto, bloco))
    else:  # troca de dois blocos (corte aleatório)
        c = np.random.randint(1, n)
        s = np.concatenate((s[c:], s[:c]))
    return s


def vns(inst, kmax=5, fator_L=1.0, tempo=5.0, semente=0, historico=False):
    np.random.seed(semente)
    _seed_numba(semente)
    args = inst.args()
    Lmax = max(10, int(fator_L * inst.n))
    s = construtiva_lpt_neh(inst)
    fs = avaliar(s, inst.n, *args)
    s, fs = busca_local(s, fs, Lmax, *args)
    hist = [(0.0, fs)]
    t0 = time.process_time()
    k = 1
    while time.process_time() - t0 < tempo:
        s1 = shaking(s, k)
        f1 = avaliar(s1, inst.n, *args)
        s2, f2 = busca_local(s1, f1, Lmax, *args)
        if f2 < fs:
            s, fs = s2, f2
            k = 1
            hist.append((time.process_time() - t0, fs))
        else:
            k = k + 1 if k < kmax else 1
    return (s, fs, hist) if historico else (s, fs)


# ---------------------------------------------------------------- ILS
@njit(cache=True)
def perturbar(seq, intensidade):
    """Segment shuffle: embaralha um trecho de tamanho intensidade*n."""
    n = len(seq)
    s = seq.copy()
    tam = max(3, int(np.ceil(intensidade * n)))
    a = np.random.randint(0, n - tam + 1)
    trecho = s[a:a + tam].copy()
    np.random.shuffle(trecho)
    s[a:a + tam] = trecho
    return s


def ils(inst, intensidade=0.1, fator_L=1.0, tempo=5.0, semente=0, historico=False):
    np.random.seed(semente)
    _seed_numba(semente)
    args = inst.args()
    Lmax = max(10, int(fator_L * inst.n))
    s = construtiva_lpt_neh(inst)
    fs = avaliar(s, inst.n, *args)
    s, fs = busca_local(s, fs, Lmax, *args)
    best, fbest = s.copy(), fs
    hist = [(0.0, fbest)]
    t0 = time.process_time()
    while time.process_time() - t0 < tempo:
        s1 = perturbar(s, intensidade)
        f1 = avaliar(s1, inst.n, *args)
        s2, f2 = busca_local(s1, f1, Lmax, *args)
        if f2 <= fs:  # aceitação better-or-equal (explora platôs)
            s, fs = s2, f2
            if fs < fbest:
                best, fbest = s.copy(), fs
                hist.append((time.process_time() - t0, fbest))
    return (best, fbest, hist) if historico else (best, fbest)


@njit(cache=True)
def _seed_numba(semente):
    np.random.seed(semente)
