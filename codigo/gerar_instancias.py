"""
Gerador de instâncias sintéticas para o problema de programação da
manutenção preventiva (PMOC) de aparelhos de ar-condicionado.

As instâncias são inspiradas na estrutura real da ALECE (4 prédios,
equipes de 2 técnicos, jornada de 9h às 17h com 1h de almoço), mas os
valores numéricos são estimados (dados reais são internos).

Formato do arquivo:
    n m H
    cap_0 cap_1 cap_2 cap_3            (vagas por prédio por dia)
    matriz 4x4 de deslocamento (min)   (diagonal = deslocamento interno)
    n linhas: id predio tipo btu p     (p = tempo de preventiva, min)
"""
import math
import os
import numpy as np

PREDIOS = ["SEDE", "ANEXO1", "ANEXO2", "ANEXO3"]
TIPOS = ["HIWALL", "PISOTETO", "CASSETE"]

# Deslocamento entre prédios (min): caminhada com ferramentas/escada.
# Diagonal: deslocamento entre salas do mesmo prédio.
DESLOC = np.array([
    [5, 10, 15, 20],
    [10, 5, 8, 15],
    [15, 8, 5, 12],
    [20, 15, 12, 5],
])

# Proporção de aparelhos por prédio e por tipo
PROP_PREDIO = [0.40, 0.25, 0.20, 0.15]
PROP_TIPO = [0.60, 0.25, 0.15]
BTUS = {
    "HIWALL": [9000, 12000, 18000, 24000],
    "PISOTETO": [24000, 36000, 48000, 60000],
    "CASSETE": [24000, 36000, 48000, 60000],
}
TEMPO_BASE = {"HIWALL": 40, "PISOTETO": 60, "CASSETE": 75}  # min (equipe de 2)

H = 420  # 9h-17h menos 1h de almoço
FOLGA_VAGAS = 1.3  # vagas/dia = 30% acima da distribuição uniforme


def tempo_preventiva(tipo, idx_btu, rng):
    """Tempo base do tipo + acréscimo por porte + dificuldade de acesso."""
    return TEMPO_BASE[tipo] + 5 * idx_btu + int(rng.integers(0, 11))


def gerar(n, m, semente, nome, pasta):
    rng = np.random.default_rng(semente)
    predio = rng.choice(4, size=n, p=PROP_PREDIO)
    linhas, tempos = [], []
    for i in range(n):
        tipo = TIPOS[rng.choice(3, p=PROP_TIPO)]
        ib = int(rng.integers(0, 4))
        p = tempo_preventiva(tipo, ib, rng)
        tempos.append(p)
        linhas.append(f"{i} {predio[i]} {tipo} {BTUS[tipo][ib]} {p}")

    # Vagas por dia em cada prédio (parâmetro que já existe no sistema).
    # Calculadas a partir de um limite inferior de dias do ciclo.
    carga = sum(tempos) + 5 * n
    dias_lb = max(1, math.ceil(carga / (m * H)))
    n_b = np.bincount(predio, minlength=4)
    cap = [max(2, math.ceil(FOLGA_VAGAS * n_b[b] / dias_lb)) for b in range(4)]

    with open(os.path.join(pasta, nome + ".txt"), "w") as f:
        f.write(f"{n} {m} {H}\n")
        f.write(" ".join(map(str, cap)) + "\n")
        for r in DESLOC:
            f.write(" ".join(map(str, r)) + "\n")
        f.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    pasta = os.path.join(os.path.dirname(__file__), "..", "instancias")
    os.makedirs(pasta, exist_ok=True)
    # Instância de referência, com porte semelhante ao caso real
    gerar(1050, 7, 2026, "ALECE-base", pasta)
    semente = 1
    for n in [1000, 1250, 1500, 2000]:
        for m in [5, 7]:
            for r in range(1, 4):
                gerar(n, m, semente, f"PMOC-n{n}-m{m}-{r}", pasta)
                semente += 1
    print("Instâncias geradas em", os.path.abspath(pasta))
