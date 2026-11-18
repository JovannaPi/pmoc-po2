"""Calcula (uma vez) a solução inicial LPT+NEH de cada instância e guarda em cache."""
import glob, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSOS
from pmoc import Instancia, construtiva_lpt_neh

def calc(c):
    t = time.time()
    construtiva_lpt_neh(Instancia(c))
    return os.path.basename(c), round(time.time() - t, 1)

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instancias")
    insts = sorted(glob.glob(os.path.join(base, "*.txt")), key=os.path.getsize, reverse=True)
    with Pool(PROCESSOS) as p:
        for r in p.imap_unordered(calc, insts):
            print(*r, flush=True)
