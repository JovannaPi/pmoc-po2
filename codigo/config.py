"""Configurações gerais do experimento."""
import os

# Tempo de CPU por execução de cada meta-heurística: TEMPO_POR_APARELHO * n segundos
TEMPO_POR_APARELHO = 0.006

# Nº de processos em paralelo. Padrão: metade dos núcleos lógicos (≈ núcleos
# físicos), para que cada execução tenha um núcleo inteiro. Pode ser alterado
# pela variável de ambiente PMOC_PROCESSOS.
PROCESSOS = int(os.environ.get("PMOC_PROCESSOS", max(1, (os.cpu_count() or 2) // 2)))
