"""
Gerador de instâncias sintéticas para o problema de programação da
manutenção preventiva (PMOC) de aparelhos de ar-condicionado.

As instâncias seguem a estrutura real da ALECE (SEDE + 4 anexos, equipes de 2
técnicos, jornada de 9h às 17h com 1h de almoço). Os tempos de preventiva vêm
de um estudo de tempos feito em campo; a composição do parque e os
deslocamentos são estimados (o cadastro do sistema é interno).

Formato do arquivo:
    n m H
    cap_0 ... cap_{B-1}                (vagas por prédio por dia; B prédios)
    matriz BxB de deslocamento (min)   (diagonal = deslocamento interno)
    n linhas: id predio tipo btu p     (p = tempo de preventiva, min)
"""
import math
import os
import numpy as np

PREDIOS = ["SEDE", "ANEXO1", "ANEXO2", "ANEXO3", "ANEXO4"]
TIPOS = ["HIWALL", "PISOTETO", "CASSETE"]

# Deslocamento entre prédios (min), estimado a partir do mapa e da experiência
# da equipe: caminhada com ferramentas e material, escadas e elevadores, e
# travessia da Av. Pontes Vieira (Anexos III e IV ficam do outro lado).
# Diagonal: troca de sala/andar no mesmo prédio (escadas e elevadores).
DESLOC = np.array([
    [8, 8, 10, 15, 20],
    [8, 8, 10, 12, 18],
    [10, 10, 8, 15, 18],
    [15, 12, 15, 8, 15],
    [20, 18, 18, 15, 8],
])

# Proporção de aparelhos por prédio e por tipo (estimadas)
PROP_PREDIO = [0.35, 0.15, 0.20, 0.15, 0.15]
PROP_TIPO = [0.60, 0.25, 0.15]
BTUS = {
    "HIWALL": [9000, 12000, 18000, 24000],
    "PISOTETO": [24000, 36000, 48000, 60000],
    "CASSETE": [24000, 36000, 48000, 60000],
}
# Tempos médios de preventiva (min, equipe de 2 técnicos), obtidos em estudo de
# tempos feito pela autora acompanhando a equipe de manutenção da ALECE:
#   split hi-wall pequeno (9-12 mil BTU) 2h; grande (18-24 mil BTU) 2h30;
#   piso-teto 3h; cassete 3h30.
TEMPO_MEDIO = {"HIWALL": [120, 120, 150, 150], "PISOTETO": [180] * 4, "CASSETE": [210] * 4}
VARIACAO_ACESSO = 10  # +-10 min conforme a dificuldade de acesso ao aparelho

H = 420  # 9h-17h menos 1h de almoço
FOLGA_VAGAS = 1.3  # vagas/dia = 30% acima da distribuição uniforme


def tempo_preventiva(tipo, idx_btu, rng):
    """Tempo médio do estudo de tempos (por tipo e porte) +- variação de acesso."""
    return TEMPO_MEDIO[tipo][idx_btu] + int(rng.integers(-VARIACAO_ACESSO, VARIACAO_ACESSO + 1))


def gerar(n, m, semente, nome, pasta):
    rng = np.random.default_rng(semente)
    predio = rng.choice(len(PREDIOS), size=n, p=PROP_PREDIO)
    linhas, tempos = [], []
    for i in range(n):
        tipo = TIPOS[rng.choice(3, p=PROP_TIPO)]
        ib = int(rng.integers(0, 4))
        p = tempo_preventiva(tipo, ib, rng)
        tempos.append(p)
        linhas.append(f"{i} {predio[i]} {tipo} {BTUS[tipo][ib]} {p}")

    # Vagas por dia em cada prédio (parâmetro que já existe no sistema).
    # Calculadas a partir de um limite inferior de dias do ciclo.
    carga = sum(tempos) + int(DESLOC.diagonal().min()) * n
    dias_lb = max(1, math.ceil(carga / (m * H)))
    n_b = np.bincount(predio, minlength=len(PREDIOS))
    cap = [max(2, math.ceil(FOLGA_VAGAS * n_b[b] / dias_lb)) for b in range(len(PREDIOS))]

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
