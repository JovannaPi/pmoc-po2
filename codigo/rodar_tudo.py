"""
Roda o experimento completo, na ordem. Pode ser interrompido (Ctrl+C, queda
de energia, computador desligado) e executado de novo: cada etapa retoma de
onde parou e execuções já salvas não são repetidas.

Uso:   python rodar_tudo.py
"""
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(AQUI, "..", "resultados")
F = os.path.join(AQUI, "..", "figuras")

ETAPAS = [
    "gerar_instancias.py",     # só roda se as instâncias não existirem
    "precalcular_neh.py",      # usa o cache já incluído no repositório
    "calibracao.py",           # retoma execuções pendentes
    "calibracao_optuna.py",    # retoma por algoritmo
    "escolher_parametros.py",
    "experimento.py",          # retoma execuções pendentes
    "analise.py",
    "curva_conclusao.py",
    "sensibilidade_equipes.py",  # 5 a 9 equipes na ALECE-base
    "figuras_extras.py",         # esquema, exemplo do decodificador, tempo do MILP
]


def main():
    os.makedirs(R, exist_ok=True)
    os.makedirs(F, exist_ok=True)
    from config import PROCESSOS, TEMPO_POR_APARELHO
    print(f"Processos em paralelo: {PROCESSOS} | tempo por execução: {TEMPO_POR_APARELHO}·n s\n")
    inicio = time.time()
    for script in ETAPAS:
        if script == "gerar_instancias.py" and os.path.exists(
                os.path.join(AQUI, "..", "instancias", "ALECE-base.txt")):
            print(f"[pulando] {script} (instâncias já existem)")
            continue
        print(f"\n=== {script} ===", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(AQUI, script)], cwd=AQUI)
        if r.returncode != 0:
            print(f"\nERRO em {script}. Rode 'python rodar_tudo.py' de novo para retomar.")
            sys.exit(1)
        print(f"[ok] {script} em {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"\nTudo pronto em {(time.time() - inicio) / 3600:.2f} h.")
    print("Agora envie os resultados:  git add -A  &&  git commit -m \"resultados\"  &&  git push")


if __name__ == "__main__":
    main()
