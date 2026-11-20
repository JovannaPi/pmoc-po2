import matplotlib.pyplot as plt

# Paleta categórica validada (slots 1-3) e cinzas para referências
COR = {"IG": "#2a78d6", "VNS": "#eb6834", "ILS": "#1baf7a", "AG": "#eda100"}
CINZA = "#8a8a83"


def aplicar_estilo():
    # Times New Roman (Windows) ou TeX Gyre Termes (clone livre do Times, Linux)
    import glob
    from matplotlib import font_manager
    for arq in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
        font_manager.fontManager.addfont(arq)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": "#55554f",
        "ytick.color": "#55554f",
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })
