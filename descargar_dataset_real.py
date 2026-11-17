"""
Descarga y unifica el dataset real de rebotes pelota-raqueta de tenis de mesa.

Fuente:
    Gossard, T. (2026). Learning Racket-Ball Bounce Dynamics Across Diverse
    Rubbers for Robotic Table Tennis. arXiv:2604.11349.
    Repositorio: https://github.com/gossardt/tt_rackets
    Licencia del repositorio: MIT.

El script descarga los 10 archivos originales (uno por configuración de raqueta),
agrega la metadata de cada configuración y calcula variables físicas derivadas.
No se inventan ni se imputan registros: solo se transforman columnas medidas.

Uso:
    python descargar_dataset_real.py
    python descargar_dataset_real.py --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_URL = "https://raw.githubusercontent.com/gossardt/tt_rackets/main"
RAW_DIR = "data/original"

# Configuraciones publicadas en el README del repositorio y en la Tabla IV del artículo.
CONFIGURACIONES = {
    1: {
        "blade": "Donic Holz Original Carbospeed",
        "blade_type": "Ofensiva",
        "sponge_mm": 2.1,
        "rubber": "andro Hexer Duro",
        "rubber_type": "Invertida ofensiva",
        "rubber_family": "Invertida",
        "rigidez_hoja": 2,
        "familia_goma": 2,
    },
    2: {
        "blade": "Donic Holz Original Carbospeed",
        "blade_type": "Ofensiva",
        "sponge_mm": 1.8,
        "rubber": "andro Power 3",
        "rubber_type": "Invertida allround",
        "rubber_family": "Invertida",
        "rigidez_hoja": 2,
        "familia_goma": 2,
    },
    3: {
        "blade": "Tibhar Holz Defense Plus",
        "blade_type": "Defensiva",
        "sponge_mm": 2.1,
        "rubber": "andro Hexer Duro",
        "rubber_type": "Invertida ofensiva",
        "rubber_family": "Invertida",
        "rigidez_hoja": 0,
        "familia_goma": 2,
    },
    4: {
        "blade": "Tibhar Holz Defense Plus",
        "blade_type": "Defensiva",
        "sponge_mm": 1.8,
        "rubber": "andro Power 3",
        "rubber_type": "Invertida allround",
        "rubber_family": "Invertida",
        "rigidez_hoja": 0,
        "familia_goma": 2,
    },
    5: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 1.2,
        "rubber": "Tibhar Grass D.TecS",
        "rubber_type": "Picos largos",
        "rubber_family": "Picos",
        "rigidez_hoja": 1,
        "familia_goma": 1,
    },
    6: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 0.0,
        "rubber": "Tibhar Grass D.TecS (OX)",
        "rubber_type": "Picos largos sin esponja",
        "rubber_family": "Picos",
        "rigidez_hoja": 1,
        "familia_goma": 1,
    },
    7: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 1.2,
        "rubber": "Dr. Neubauer Diamant",
        "rubber_type": "Picos medios",
        "rubber_family": "Picos",
        "rigidez_hoja": 1,
        "familia_goma": 1,
    },
    8: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 2.0,
        "rubber": "andro Blowfish",
        "rubber_type": "Picos cortos",
        "rubber_family": "Picos",
        "rigidez_hoja": 1,
        "familia_goma": 1,
    },
    9: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 2.1,
        "rubber": "andro Hexer Duro",
        "rubber_type": "Invertida ofensiva",
        "rubber_family": "Invertida",
        "rigidez_hoja": 1,
        "familia_goma": 2,
    },
    10: {
        "blade": "DONIC Appelgren Allplay Senso V1",
        "blade_type": "Allrounder",
        "sponge_mm": 2.1,
        "rubber": "Dr. Neubauer A-B-S II soft",
        "rubber_type": "Antispin",
        "rubber_family": "Antispin",
        "rigidez_hoja": 1,
        "familia_goma": 0,
    },
}


def descargar(url: str, destino: Path, force: bool = False) -> Path:
    if destino.exists() and not force:
        print(f"   Ya existe: {destino.name}")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "proyecto-ml-tenis-mesa/1.0"})
    with urlopen(req, timeout=120) as respuesta:
        contenido = respuesta.read()
    destino.write_bytes(contenido)
    print(f"   Descargado: {destino.name} ({len(contenido) / 1024:.0f} KB)")
    return destino


def agregar_variables_fisicas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Velocidades normal y tangencial. En el dataset v_in_z es negativa porque
    # la pelota llega hacia la raqueta; se usa su valor absoluto como magnitud.
    out["v_in_norm"] = out["v_in_z"].abs()
    out["v_in_tan"] = out["v_in_x"]
    out["v_out_norm"] = out["v_out_z"]
    out["v_out_tan"] = out["v_out_x"]
    out["spin_in"] = out["spin_before_y"]
    out["spin_out"] = out["spin_after_y"]

    out["speed_in"] = np.sqrt(out["v_in_x"] ** 2 + out["v_in_z"] ** 2)
    out["speed_out"] = np.sqrt(out["v_out_x"] ** 2 + out["v_out_z"] ** 2)
    out["cor"] = out["v_out_norm"] / out["v_in_norm"]
    out["energy_ratio"] = (out["speed_out"] ** 2) / (out["speed_in"] ** 2)
    out["delta_spin"] = out["spin_out"] - out["spin_in"]
    out["angulo_incidencia"] = np.arctan2(out["v_in_x"], out["v_in_norm"])
    out["spin_relativo"] = out["spin_in"] / out["v_in_norm"]
    out["ratio_tan_norm"] = out["v_in_tan"] / out["v_in_norm"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset real de tt_rackets.")
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar los CSV originales.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "datos" / "original"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  DESCARGA DEL DATASET REAL - TT_RACKETS (MIT)")
    print("=" * 78)
    print(f"Fuente: {BASE_URL}/{RAW_DIR}")
    print(f"Destino: {raw_dir}\n")

    partes = []
    for racket_id, meta in CONFIGURACIONES.items():
        nombre = f"racket_{racket_id:02d}.csv"
        url = f"{BASE_URL}/{RAW_DIR}/{nombre}"
        ruta = descargar(url, raw_dir / nombre, force=args.force)
        df = pd.read_csv(ruta)
        df.insert(0, "racket_id", racket_id)
        for clave, valor in meta.items():
            df[clave] = valor
        partes.append(df)
        print(f"   Racket {racket_id:02d}: {len(df):4d} rebotes | "
              f"{meta['blade_type']:10s} | {meta['sponge_mm']:.1f} mm | "
              f"{meta['rubber_type']}")

    df = pd.concat(partes, ignore_index=True)
    df = agregar_variables_fisicas(df)

    # Columna legible para agrupar y para las tablas del notebook.
    df["configuracion"] = df.apply(
        lambda r: (f"R{int(r['racket_id']):02d} - {r['blade_type']} / "
                   f"{r['sponge_mm']:.1f} mm / {r['rubber_type']}"),
        axis=1,
    )

    # Orden de columnas: identificación, metadata, medidas originales y derivadas.
    columnas_orden = [
        "racket_id", "configuracion", "blade", "blade_type", "rigidez_hoja",
        "sponge_mm", "rubber", "rubber_type", "rubber_family", "familia_goma",
        "racket_angle", "traj_dir",
        "v_in_x", "v_in_y", "v_in_z", "v_out_x", "v_out_y", "v_out_z",
        "spin_before_x", "spin_before_y", "spin_before_z",
        "spin_after_x", "spin_after_y", "spin_after_z",
        "v_in_norm", "v_in_tan", "v_out_norm", "v_out_tan",
        "spin_in", "spin_out", "speed_in", "speed_out", "cor",
        "energy_ratio", "delta_spin", "angulo_incidencia", "spin_relativo",
        "ratio_tan_norm",
    ]
    df = df[columnas_orden]

    salida = base_dir / "datos" / "dataset_rebotes_tt.csv"
    df.to_csv(salida, index=False, encoding="utf-8")

    resumen = {
        "fuente": "https://github.com/gossardt/tt_rackets",
        "articulo": "https://arxiv.org/abs/2604.11349",
        "licencia": "MIT",
        "fecha_descarga": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archivos_originales": 10,
        "filas": int(len(df)),
        "columnas": list(df.columns),
        "configuraciones": int(df["racket_id"].nunique()),
        "rebotes_por_configuracion": {
            str(k): int(v) for k, v in df["racket_id"].value_counts().sort_index().items()
        },
    }
    with open(base_dir / "datos" / "resumen_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"  DATASET UNIFICADO: {len(df)} rebotes x {len(df.columns)} columnas")
    print(f"  Archivo: {salida}")
    print(f"  Configuraciones: {df['racket_id'].nunique()}")
    print("=" * 78)
    print(df.groupby(["racket_id", "rubber_type"]).size().to_string())


if __name__ == "__main__":
    main()
