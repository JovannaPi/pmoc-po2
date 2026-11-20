# PMOC — Programação da manutenção preventiva de ar-condicionado

Projeto de Pesquisa Operacional II (UFC, Prof. Bruno de Athayde Prata).
Autora: Jovanna Pinheiro Ferreira.

Equipes de manutenção são tratadas como máquinas paralelas idênticas, com
tempo de setup dependente da sequência (deslocamento entre prédios), jornada
diária e limite de vagas por prédio por dia. Objetivo: minimizar o makespan
(Cmax). Métodos: regra atual do sistema, LPT+NEH, IG, VNS, ILS e algoritmo genético
(AG). Um modelo MILP (Python-MIP + CBC) valida as meta-heurísticas em
instâncias pequenas (`codigo/modelo_exato.py`).

## Como rodar (Windows, pelo VS Code)

**Antes (uma vez só):**
1. Instale o **Python 3.11 ou mais novo** em https://www.python.org/downloads/
   e marque **"Add python.exe to PATH"** na instalação.
2. Instale o **Git** em https://git-scm.com/download/win (opções padrão).
3. **Evite a suspensão:** Configurações → Sistema → Energia → "Quando conectado,
   colocar o dispositivo em suspensão após" = **Nunca**. Em notebook, também em
   Painel de Controle → Opções de Energia → "Escolher a função do fechamento da
   tampa" → **Nada a fazer** (na tomada). Deixe o carregador ligado.

**Rodar:**
1. Abra o VS Code → menu **Terminal → Novo Terminal**.
2. No terminal, rode (se você já clonou antes, pule o `git clone` e faça só
   `cd pmoc-po2` e `git pull`):

```bat
git clone https://github.com/JovannaPi/pmoc-po2.git
cd pmoc-po2
pip install -r requirements.txt
cd codigo
python rodar_tudo.py
```

3. Pode minimizar o VS Code, mas **não feche** o VS Code nem o terminal. Tempo
   estimado: 3 a 6 h, conforme o processador. O andamento aparece no terminal.
4. Se cair ou você fechar sem querer: abra o terminal de novo, `cd pmoc-po2\codigo`
   e `python rodar_tudo.py`. Ele **continua de onde parou**.
5. Quando aparecer **"Tudo pronto"**, envie os resultados:

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
| 7. Análise estatística | `analise.py` | `resultados/analise_estatistica.txt` (Shapiro-Wilk, Kruskal-Wallis + Dunn, Friedman + Nemenyi), boxplot, Q-Q plot, convergência por tempo e por iteração, RPD por tamanho, histórico do Optuna |
| 8. Estudo de caso | `curva_conclusao.py` | `figuras/curva_conclusao_alece.*`, `figuras/gantt_alece_dias1e2.*` |
| 9. Sensibilidade | `sensibilidade_equipes.py` | dias do ciclo com 5 a 9 equipes (`figuras/sensibilidade_equipes.*`) |
| 10. Figuras de apoio | `figuras_extras.py` | esquema do problema, exemplo do decodificador, tempo do MILP |
| (à parte) Validação exata | `modelo_exato.py` | `resultados/validacao_exata.csv` (MILP × meta-heurísticas, 16 instâncias pequenas); rodada depois do experimento, com os parâmetros calibrados |

## Instâncias

25 instâncias sintéticas: `ALECE-base` (1050 aparelhos, 7 equipes) e
`PMOC-n{1000,1250,1500,2000}-m{5,7}-{1,2,3}`. A estrutura vem do caso real
(SEDE + 4 anexos, equipes de 2 técnicos das 9h às 17h, tipos hi-wall, piso-teto e
cassete). Os tempos de preventiva vêm de um estudo de tempos feito em campo pela autora
(hi-wall pequeno 2h, hi-wall grande 2h30, piso-teto 3h, cassete 3h30, ±10 min de
variação de acesso). A composição do parque e os deslocamentos são estimados,
porque o cadastro do sistema é interno. Resultados da versão anterior, com
tempos estimados, estão em `arquivo/v1-tempos-estimados/`.

Formato:

```
n m H
c_SEDE c_A1 c_A2 c_A3 c_A4      (vagas por prédio por dia)
matriz 5x5 de deslocamento (min)
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
