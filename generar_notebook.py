"""
Convierte el script principal en un notebook de Jupyter/Colab.

El script usa marcadores `# %%` y `# %% [markdown]`, el mismo formato que
Jupytext. Este generador no necesita Jupytext: construye el notebook con
nbformat y conserva el orden de las celdas.

Uso:
    python generar_notebook.py
"""

from pathlib import Path

import nbformat as nbf


BASE_DIR = Path(__file__).resolve().parent
RUTA_SCRIPT = BASE_DIR / "proyecto_ml_tenis_mesa_corregido.py"
RUTA_NOTEBOOK = BASE_DIR / "proyecto_ml_tenis_mesa_corregido.ipynb"


def dividir_en_celdas(texto: str):
    celdas = []
    tipo = "code"
    buffer = []

    def cerrar():
        no_vacias = [linea for linea in buffer if linea.strip()]
        if no_vacias:
            celdas.append((tipo, "\n".join(buffer).strip("\n")))

    for linea in texto.splitlines():
        if linea.startswith("# %%"):
            cerrar()
            buffer = []
            tipo = "markdown" if "[markdown]" in linea else "code"
            continue
        buffer.append(linea)
    cerrar()
    return celdas


def main() -> None:
    texto = RUTA_SCRIPT.read_text(encoding="utf-8")
    celdas = dividir_en_celdas(texto)

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
        "colab": {"provenance": []},
    }

    nb["cells"].append(nbf.v4.new_markdown_cell(
        "# Proyecto ML - Rebotes de tenis de mesa con datos reales\n\n"
        "**Fuente:** Gossard, T. (2026). *Learning Racket-Ball Bounce Dynamics "
        "Across Diverse Rubbers for Robotic Table Tennis*. arXiv:2604.11349. "
        "Repositorio: https://github.com/gossardt/tt_rackets (MIT).\n\n"
        "Este notebook compara 11 clasificadores y 11 regresores sobre 8.184 "
        "rebotes reales medidos en 10 configuraciones de raqueta.\n\n"
        "**En Google Colab:** ejecuta la celda de instalación y, si el CSV no "
        "está en el repositorio, ejecuta `python descargar_dataset_real.py`."
    ))

    nb["cells"].append(nbf.v4.new_code_cell(
        "import sys\n"
        "if 'google.colab' in sys.modules:\n"
        "    import subprocess\n"
        "    subprocess.run(\n"
        "        [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],\n"
        "        check=True,\n"
        "    )"
    ))

    for tipo, contenido in celdas:
        if tipo == "markdown":
            lineas = []
            for linea in contenido.splitlines():
                if linea.startswith("# "):
                    lineas.append(linea[2:])
                elif linea.strip() == "#":
                    lineas.append("")
                else:
                    lineas.append(linea)
            nb["cells"].append(nbf.v4.new_markdown_cell("\n".join(lineas).strip()))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(contenido))

    with open(RUTA_NOTEBOOK, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook generado: {RUTA_NOTEBOOK}")
    print(f"Celdas: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
