"""
Genera tablas Markdown a partir de los resultados exportados por el pipeline.

Uso:
    python resumir_resultados.py
"""

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELOS = BASE_DIR / "modelos"


def tabla_clf(df: pd.DataFrame) -> str:
    filas = ["| Modelo | F1 CV | F1 Test | Accuracy Test |", "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        filas.append(
            f"| {r['Modelo']} | {r['F1_CV']:.4f} | {r['F1_Test']:.4f} | "
            f"{r['Accuracy_Test']:.4f} |"
        )
    return "\n".join(filas)


def tabla_reg(df: pd.DataFrame) -> str:
    filas = ["| Modelo | R2 CV | R2 Test | RMSE Test |", "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        filas.append(
            f"| {r['Modelo']} | {r['R2_CV']:.4f} | {r['R2_Test']:.4f} | "
            f"{r['RMSE_Test']:.4f} |"
        )
    return "\n".join(filas)


def main() -> None:
    clf = pd.read_csv(MODELOS / "resultados_clasificacion.csv")
    reg = pd.read_csv(MODELOS / "resultados_regresion.csv")

    print("## Clasificación\n")
    print(tabla_clf(clf))
    print("\n## Regresión\n")
    print(tabla_reg(reg))

    ruta_meta = MODELOS / "metadata.json"
    if ruta_meta.exists():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        print("\n## Diagnóstico de fuga\n")
        diag = meta.get("diagnostico_fuga", {})
        for clave, valor in diag.items():
            print(f"- `{clave}`: {valor}")
        print("\n## Validación ONNX\n")
        for clave, valor in meta.get("validacion_onnx", {}).items():
            print(f"- `{clave}`: {valor}")


if __name__ == "__main__":
    main()
