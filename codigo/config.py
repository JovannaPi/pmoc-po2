"""Configurações gerais do experimento."""
import os

# Tempo de CPU por execução de cada meta-heurística (segundos por aparelho):
#  - calibração: TEMPO_POR_APARELHO * n (tempo reduzido, por custo computacional);
#  - experimento final e sensibilidade: TEMPO_EXPERIMENTO * n. O tempo maior foi
#    adotado porque, com 0,006n s, as curvas de convergência ainda caíam ao fim
#    do tempo (resultados dessa rodada em arquivo/v2-tempo-curto/).
TEMPO_POR_APARELHO = 0.006
TEMPO_EXPERIMENTO = 0.02

# Nº de processos em paralelo. Padrão: metade dos núcleos lógicos (≈ núcleos
# físicos), para que cada execução tenha um núcleo inteiro. Pode ser alterado
# pela variável de ambiente PMOC_PROCESSOS.
PROCESSOS = int(os.environ.get("PMOC_PROCESSOS", max(1, (os.cpu_count() or 2) // 2)))
