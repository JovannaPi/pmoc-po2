# PMOC — Programação da manutenção preventiva de ar-condicionado

Projeto de Pesquisa Operacional II (UFC, Prof. Bruno de Athayde Prata).
Autora: Jovanna Pinheiro Ferreira.

Equipes de manutenção são tratadas como máquinas paralelas idênticas, com
tempo de setup dependente da sequência (deslocamento entre prédios), jornada
diária e limite de vagas por prédio por dia. Objetivo: minimizar o makespan
(Cmax). Métodos: regra atual do sistema, LPT+NEH, IG, VNS, ILS e algoritmo genético
(AG). Um modelo MILP (Python-MIP + CBC) valida as meta-heurísticas em
instâncias pequenas (`codigo/modelo_exato.py`).

## Como rodar (Windows)

1. Instale o **Python 3.11 ou mais novo** em https://www.python.org/downloads/
   (na instalação, marque **"Add python.exe to PATH"**).
2. Instale o **Git** em https://git-scm.com/download/win (pode aceitar as opções padrão).
3. Abra o **Prompt de Comando** (tecla Windows → digite `cmd` → Enter) e rode:

```bat
git clone https://github.com/JovannaPi/pmoc-po2.git
cd pmoc-po2
pip install -r requirements.txt
cd codigo
python rodar_tudo.py
```

4. Deixe o computador ligado e na tomada, e configure-o para **não entrar em
   suspensão** durante a execução (Configurações → Sistema → Energia).
   Tempo estimado: 2 a 4 h, dependendo do número de núcleos do processador.
5. Se cair, travar ou você precisar desligar: abra o Prompt de Comando de novo,
   entre na pasta (`cd pmoc-po2\codigo`) e rode `python rodar_tudo.py`.
   Ele **continua de onde parou**.
6. Quando terminar (mensagem "Tudo pronto"), envie os resultados:

```bat
cd ..
git add -A
git commit -m "resultados do experimento"
git pull --no-edit
git push
```

> No Mac/Linux os comandos são os mesmos (use `python3` e `pip3` se
> `python` não funcionar).

### Ajustes opcionais

- Número de processos em paralelo: por padrão, metade dos núcleos lógicos.
  Para mudar: `set PMOC_PROCESSOS=4` (Windows) antes de `python rodar_tudo.py`.
  Não use mais processos que núcleos físicos: o tempo limite é medido em tempo
  de CPU, e núcleos compartilhados deixam cada execução mais lenta.
- Tempo por execução: `TEMPO_POR_APARELHO` em `codigo/config.py`.

## Etapas (feitas automaticamente por `rodar_tudo.py`)

| Etapa | Script | Saída |
|---|---|---|
| 1. Instâncias | `gerar_instancias.py` | `instancias/*.txt` (já incluídas) |
| 2. Solução inicial | `precalcular_neh.py` | `instancias/neh/*.npy` (cache da LPT+NEH, já incluído) |
| 3. Calibração por grade (aula) | `calibracao.py` | `resultados/calibracao_*.csv` |
| 4. Refinamento com Optuna | `calibracao_optuna.py` | `resultados/optuna_*` |
| 5. Escolha dos parâmetros | `escolher_parametros.py` | `resultados/parametros_finais.json`, `figuras/calibracao_boxplot.*` |
| 6. Experimento final (30 execuções) | `experimento.py` | `resultados/experimento_execucoes.csv`, `convergencia.csv`, `deterministicos.csv` |
| 7. Análise estatística | `analise.py` | `resultados/analise_estatistica.txt`, boxplot, Q-Q plot, convergência |
| 8. Estudo de caso | `curva_conclusao.py` | `figuras/curva_conclusao_alece.*` |

## Instâncias

25 instâncias sintéticas: `ALECE-base` (1050 aparelhos, 7 equipes) e
`PMOC-n{1000,1250,1500,2000}-m{5,7}-{1,2,3}`. A estrutura vem do caso real
(4 prédios, equipes de 2 técnicos das 9h às 17h, tipos hi-wall, piso-teto e
cassete). Os valores são estimados, porque os dados do sistema são internos.

Formato:

```
n m H
c_SEDE c_A1 c_A2 c_A3          (vagas por prédio por dia)
matriz 4x4 de deslocamento (min)
id predio tipo btu p           (uma linha por aparelho; p em minutos)
```

## Arquivos de código

| Arquivo | Descrição |
|---|---|
| `pmoc.py` | Instância, decodificador (list scheduling), avaliação acelerada, regra atual, LPT+NEH |
| `metaheuristicas.py` | IG, VNS, ILS e AG |
| `modelo_exato.py` | Modelo MILP e validação em instâncias pequenas (opcional; já rodado) |
| `config.py` | Tempo por execução e nº de processos |
| `estilo.py`, `gantt.py` | Estilo dos gráficos e reconstrução da programação detalhada |
