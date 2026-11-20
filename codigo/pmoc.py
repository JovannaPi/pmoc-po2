"""
Núcleo do problema: leitura da instância, decodificador (list scheduling)
e heurísticas construtivas.

Representação: uma permutação dos aparelhos. O decodificador percorre a
permutação e aloca cada aparelho à equipe que consegue concluí-lo mais cedo,
respeitando (i) a jornada diária H, (ii) o deslocamento dependente da
sequência entre prédios (tempo de setup) e (iii) o limite de vagas por
prédio por dia. Cada equipe inicia o dia na SEDE.
"""
import numpy as np
from numba import njit

BIG = 10 ** 12


class Instancia:
    def __init__(self, caminho):
        self.caminho = caminho
        with open(caminho) as f:
            linhas = [l.split() for l in f if l.strip()]
        self.n, self.m, self.H = map(int, linhas[0])
        self.m_arquivo = self.m
        self.cap = np.array(list(map(int, linhas[1])), dtype=np.int64)
        self.T = np.array([list(map(int, l)) for l in linhas[2:6]], dtype=np.int64)
        dados = linhas[6:]
        self.predio = np.array([int(l[1]) for l in dados], dtype=np.int64)
        self.tipo = [l[2] for l in dados]
        self.btu = [int(l[3]) for l in dados]
        self.p = np.array([int(l[4]) for l in dados], dtype=np.int64)
        self.maxD = self.n + 2

    def args(self):
        return (self.predio, self.p, self.T, self.cap, self.m, self.H, self.maxD)


@njit(cache=True)
def decodificar(seq, L, predio, p, T, cap, m, H, maxD):
    """Retorna (Cmax em minutos úteis, soma dos términos, deslocamento total, nº de dias)."""
    dia = np.zeros(m, np.int64)
    t = np.zeros(m, np.int64)
    loc = np.zeros(m, np.int64)
    uso = np.zeros((maxD, 4), np.int64)
    desloc = 0
    for idx in range(L):
        j = seq[idx]
        b = predio[j]
        melhor = BIG
        bk = 0
        bd = 0
        bfim = 0
        bdes = 0
        for k in range(m):
            d = dia[k]
            des = T[loc[k], b]
            fim = t[k] + des + p[j]
            if not (fim <= H and uso[d, b] < cap[b]):
                d += 1
                while uso[d, b] >= cap[b]:
                    d += 1
                des = T[0, b]
                fim = des + p[j]
            ab = d * H + fim
            if ab < melhor:
                melhor = ab
                bk = k
                bd = d
                bfim = fim
                bdes = des
        dia[bk] = bd
        t[bk] = bfim
        loc[bk] = b
        uso[bd, b] += 1
        desloc += bdes
    cmax = 0
    soma = 0
    for k in range(m):
        ab = dia[k] * H + t[k]
        soma += ab
        if ab > cmax:
            cmax = ab
    ndias = (cmax - 1) // H + 1 if cmax > 0 else 0
    return cmax, soma, desloc, ndias


@njit(cache=True)
def avaliar(seq, L, predio, p, T, cap, m, H, maxD):
    """Função de avaliação: Cmax com desempate pela soma dos términos."""
    c, s, _, _ = decodificar(seq, L, predio, p, T, cap, m, H, maxD)
    return c + 1e-7 * s


@njit(cache=True)
def melhor_insercao(parcial, L, j, buf, predio, p, T, cap, m, H, maxD):
    """Insere j em todas as L+1 posições de parcial[:L]; devolve (pos, f)."""
    melhor_f = 1e18
    melhor_pos = 0
    for pos in range(L + 1):
        for i in range(pos):
            buf[i] = parcial[i]
        buf[pos] = j
        for i in range(pos, L):
            buf[i + 1] = parcial[i]
        f = avaliar(buf, L + 1, predio, p, T, cap, m, H, maxD)
        if f < melhor_f:
            melhor_f = f
            melhor_pos = pos
    return melhor_pos, melhor_f


@njit(cache=True)
def inserir(parcial, L, j, pos):
    for i in range(L, pos, -1):
        parcial[i] = parcial[i - 1]
    parcial[pos] = j


@njit(cache=True)
def neh(ordem, predio, p, T, cap, m, H, maxD):
    """Heurística de inserção NEH a partir de uma ordem inicial."""
    n = len(ordem)
    seq = np.empty(n, np.int64)
    buf = np.empty(n, np.int64)
    seq[0] = ordem[0]
    L = 1
    for idx in range(1, n):
        j = ordem[idx]
        pos, _ = melhor_insercao_rapida(seq, L, j, predio, p, T, cap, m, H, maxD)
        inserir(seq, L, j, pos)
        L += 1
    return seq


def plano_regra_atual(inst):
    """Regra do sistema atual: em cada prédio, os aparelhos (na ordem do
    cadastro) ocupam as vagas diárias do prédio, dia após dia. Todo aparelho
    'vale' o mesmo e o deslocamento é ignorado. Devolve o dia planejado de
    cada aparelho."""
    dia_plan = np.zeros(inst.n, dtype=np.int64)
    cont = np.zeros(4, dtype=np.int64)
    for j in range(inst.n):
        b = inst.predio[j]
        dia_plan[j] = cont[b] // inst.cap[b]
        cont[b] += 1
    return dia_plan


def regra_atual(inst):
    """Sequência executada pela regra atual: dia planejado, depois prédio,
    depois ordem de cadastro. A execução real é avaliada pelo decodificador
    (tempos reais e deslocamentos)."""
    dp = plano_regra_atual(inst)
    return np.array(sorted(range(inst.n), key=lambda j: (dp[j], inst.predio[j], j)), dtype=np.int64)


def dias_planejados(inst):
    """Dias que o modelo simplificado do sistema promete para o ciclo."""
    return int(plano_regra_atual(inst).max() + 1)


def construtiva_lpt_neh(inst):
    """Solução inicial: ordem LPT (maior tempo primeiro) refinada pela NEH.
    É determinística; o resultado fica em cache (instancias/neh/) para não
    ser recalculado em cada execução."""
    import os
    pasta = os.path.join(os.path.dirname(inst.caminho), "neh")
    sufixo = "" if inst.m == inst.m_arquivo else f"_m{inst.m}"
    arq = os.path.join(pasta, os.path.basename(inst.caminho)[:-4] + sufixo + ".npy")
    if os.path.exists(arq):
        return np.load(arq)
    ordem = np.argsort(-inst.p, kind="stable").astype(np.int64)
    seq = neh(ordem, *inst.args())
    os.makedirs(pasta, exist_ok=True)
    np.save(arq, seq)
    return seq


def limite_inferior(inst):
    """Limite inferior simples do Cmax: carga mínima (tempo + 5 min de
    deslocamento interno por aparelho) dividida igualmente entre equipes,
    considerando a perda de fim de dia nula."""
    return int(np.ceil((inst.p.sum() + 5 * inst.n) / inst.m))


# ------------------------------------------------------------------
# Avaliação acelerada (mesmo resultado, menos operações)
# ------------------------------------------------------------------
@njit(cache=True)
def _alocar(j, dia, t, loc, uso, predio, p, T, cap, m, H):
    """Regra do decodificador para um aparelho; devolve (equipe, dia, fim, ab)."""
    b = predio[j]
    melhor = BIG
    bk = 0
    bd = 0
    bfim = 0
    for k in range(m):
        d = dia[k]
        fim = t[k] + T[loc[k], b] + p[j]
        if not (fim <= H and uso[d, b] < cap[b]):
            d += 1
            while uso[d, b] >= cap[b]:
                d += 1
            fim = T[0, b] + p[j]
        ab = d * H + fim
        if ab < melhor:
            melhor = ab
            bk = k
            bd = d
            bfim = fim
    return bk, bd, bfim, melhor


@njit(cache=True)
def avaliar_limite(seq, L, lim, predio, p, T, cap, m, H, maxD):
    """Como avaliar(), mas interrompe (devolve 1e18) assim que o Cmax parcial
    passa de lim. Como o Cmax nunca diminui ao alocar mais aparelhos, uma
    solução interrompida é certamente pior que lim."""
    dia = np.zeros(m, np.int64)
    t = np.zeros(m, np.int64)
    loc = np.zeros(m, np.int64)
    uso = np.zeros((maxD, 4), np.int64)
    cmax = 0
    for idx in range(L):
        j = seq[idx]
        k, d, fim, ab = _alocar(j, dia, t, loc, uso, predio, p, T, cap, m, H)
        dia[k] = d
        t[k] = fim
        loc[k] = predio[j]
        uso[d, predio[j]] += 1
        if ab > cmax:
            cmax = ab
            if cmax > lim:
                return 1e18
    soma = 0
    for k in range(m):
        soma += dia[k] * H + t[k]
    return cmax + 1e-7 * soma


@njit(cache=True)
def melhor_insercao_rapida(parcial, L, j, predio, p, T, cap, m, H, maxD):
    """Mesmo resultado de melhor_insercao(), reaproveitando o estado do
    prefixo comum e interrompendo posições cujo Cmax parcial já supera o
    melhor encontrado."""
    dia = np.zeros(m, np.int64)
    t = np.zeros(m, np.int64)
    loc = np.zeros(m, np.int64)
    uso = np.zeros((maxD, 4), np.int64)
    dd = np.empty(m, np.int64)
    tt = np.empty(m, np.int64)
    ll = np.empty(m, np.int64)
    log_d = np.empty(L + 1, np.int64)
    log_b = np.empty(L + 1, np.int64)
    melhor_f = 1e18
    melhor_c = BIG
    melhor_pos = 0
    cmax_pref = 0
    for pos in range(L + 1):
        for k in range(m):
            dd[k] = dia[k]
            tt[k] = t[k]
            ll[k] = loc[k]
        cmax = cmax_pref
        nlog = 0
        cortado = False
        for idx in range(pos - 1, L):
            jj = j if idx == pos - 1 else parcial[idx]
            k, d, fim, ab = _alocar(jj, dd, tt, ll, uso, predio, p, T, cap, m, H)
            dd[k] = d
            tt[k] = fim
            ll[k] = predio[jj]
            uso[d, predio[jj]] += 1
            log_d[nlog] = d
            log_b[nlog] = predio[jj]
            nlog += 1
            if ab > cmax:
                cmax = ab
                if cmax > melhor_c:
                    cortado = True
                    break
        for i in range(nlog):
            uso[log_d[i], log_b[i]] -= 1
        if not cortado:
            soma = 0
            for k in range(m):
                soma += dd[k] * H + tt[k]
            f = cmax + 1e-7 * soma
            if f < melhor_f:
                melhor_f = f
                melhor_c = cmax
                melhor_pos = pos
        # avança o prefixo com parcial[pos]
        if pos < L:
            jj = parcial[pos]
            k, d, fim, ab = _alocar(jj, dia, t, loc, uso, predio, p, T, cap, m, H)
            dia[k] = d
            t[k] = fim
            loc[k] = predio[jj]
            uso[d, predio[jj]] += 1
            if ab > cmax_pref:
                cmax_pref = ab
    return melhor_pos, melhor_f
