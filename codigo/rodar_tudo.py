"""
Roda o experimento completo, na ordem. Pode ser interrompido (Ctrl+C, queda
de energia, computador desligado) e executado de novo: as etapas já
concluídas são puladas e as execuções já salvas não são repetidas.

Uso:   python rodar_tudo.py
"""
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(AQUI, "..", "resultados")
F = os.path.join(AQUI, "..", "figuras")

# (script, arquivo que indica que a etapa já terminou)
ETAPAS = [
    ("gerar_instancias.py", os.path.join(AQUI, "..", "instancias", "ALECE-base.txt")),
    ("precalcular_neh.py", os.path.join(AQUI, "..", "instancias", "neh", "ALECE-base.npy")),
    ("calibracao.py", os.path.join(R, "calibracao_resumo.csv")),
    ("calibracao_optuna.py", os.path.join(R, "optuna_melhores.json")),
    ("escolher_parametros.py", os.path.join(R, "parametros_finais.json")),
    ("experimento.py", os.path.join(R, "deterministicos.csv")),
    ("analise.py", os.path.join(R, "analise_estatistica.txt")),
    ("curva_conclusao.py", os.path.join(F, "curva_conclusao_alece.pdf")),
]


def main():
    os.makedirs(R, exist_ok=True)
    os.makedirs(F, exist_ok=True)
    from config import PROCESSOS, TEMPO_POR_APARELHO
    print(f"Processos em paralelo: {PROCESSOS} | tempo por execução: {TEMPO_POR_APARELHO}·n s\n")
    inicio = time.time()
    for script, pronto in ETAPAS:
        # O experimento só é considerado pronto quando todas as execuções terminaram
        if script == "experimento.py" and os.path.exists(os.path.join(R, "experimento_completo.txt")):
            print(f"[pulando] {script} (já concluído)")
            continue
        if script != "experimento.py" and os.path.exists(pronto):
            print(f"[pulando] {script} (já concluído)")
            continue
        print(f"\n=== {script} ===", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(AQUI, script)], cwd=AQUI)
        if r.returncode != 0:
            print(f"\nERRO em {script}. Rode 'python rodar_tudo.py' de novo para retomar.")
            sys.exit(1)
        if script == "experimento.py":
            with open(os.path.join(R, "experimento_completo.txt"), "w") as f:
                f.write("ok\n")
        print(f"[ok] {script} em {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"\nTudo pronto em {(time.time() - inicio) / 3600:.2f} h.")
    print("Agora envie os resultados:  git add -A  &&  git commit -m \"resultados\"  &&  git push")


if __name__ == "__main__":
    main()
