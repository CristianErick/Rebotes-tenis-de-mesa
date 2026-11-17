# Fuente del dataset real

## Referencia

Thomas Gossard. **Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers
for Robotic Table Tennis.** arXiv:2604.11349, 2026.

- Artículo: https://arxiv.org/abs/2604.11349
- Repositorio: https://github.com/gossardt/tt_rackets
- Licencia del repositorio: **MIT** (ver `LICENSE_TT_RACKETS.txt`).

## Qué contiene

El repositorio publica 8.184 rebotes reales de pelota contra 10 configuraciones
de raqueta. Cada archivo original (`datos/original/racket_XX.csv`) corresponde
a una configuración y contiene, por rebote:

- ángulo de la raqueta (`racket_angle`);
- dirección de la trayectoria (`traj_dir`);
- componentes del spin antes y después del rebote
  (`spin_before_*`, `spin_after_*`);
- componentes de la velocidad antes y después del rebote
  (`v_in_*`, `v_out_*`).

Las unidades del repositorio son metros por segundo para las velocidades y
radianes por segundo para el spin. El movimiento está restringido a un plano
2D, por lo que `v_in_y` y `v_out_y` son cero.

## Configuraciones

| ID | Hoja | Espesor de esponja | Goma | Tipo de goma |
|---:|---|---:|---|---|
| 1 | Ofensiva | 2.1 mm | andro Hexer Duro | Invertida ofensiva |
| 2 | Ofensiva | 1.8 mm | andro Power 3 | Invertida allround |
| 3 | Defensiva | 2.1 mm | andro Hexer Duro | Invertida ofensiva |
| 4 | Defensiva | 1.8 mm | andro Power 3 | Invertida allround |
| 5 | Allrounder | 1.2 mm | Tibhar Grass D.TecS | Picos largos |
| 6 | Allrounder | 0.0 mm | Tibhar Grass D.TecS (OX) | Picos largos sin esponja |
| 7 | Allrounder | 1.2 mm | Dr. Neubauer Diamant | Picos medios |
| 8 | Allrounder | 2.0 mm | andro Blowfish | Picos cortos |
| 9 | Allrounder | 2.1 mm | andro Hexer Duro | Invertida ofensiva |
| 10 | Allrounder | 2.1 mm | Dr. Neubauer A-B-S II soft | Antispin |

## Variables derivadas

El archivo unificado `dataset_rebotes_tt.csv` conserva las columnas originales
y agrega variables calculadas, no inventadas:

- `v_in_norm = |v_in_z|`: velocidad normal de entrada.
- `v_in_tan = v_in_x`: velocidad tangencial de entrada.
- `v_out_norm = v_out_z`: velocidad normal de salida.
- `v_out_tan = v_out_x`: velocidad tangencial de salida.
- `spin_in = spin_before_y`, `spin_out = spin_after_y`: spin principal.
- `cor = v_out_norm / v_in_norm`: coeficiente de restitución del rebote.
- `energy_ratio = speed_out^2 / speed_in^2`: relación de energía cinética.
- `delta_spin = spin_out - spin_in`: cambio de spin.
- `angulo_incidencia = atan2(v_in_x, v_in_norm)`.
- `spin_relativo = spin_in / v_in_norm`.
- `ratio_tan_norm = v_in_tan / v_in_norm`.

La metadata de la raqueta (`blade`, `blade_type`, `sponge_mm`, `rubber`,
`rubber_type`, `rubber_family`) se toma de la Tabla IV del artículo y del README
del repositorio.

## Cómo se obtuvo

El script `descargar_dataset_real.py` descarga los 10 CSV originales desde
GitHub, verifica que se puedan leer, agrega la metadata de cada configuración,
calcula las variables derivadas y genera:

- `datos/dataset_rebotes_tt.csv` (dataset unificado);
- `datos/resumen_dataset.json` (trazabilidad y conteos).

No se rasparon sitios web ni se usaron datos sintéticos. Los archivos originales
se conservan en `datos/original/` para reproducibilidad.

## Limitaciones que deben declararse

- Solo hay 10 configuraciones de raqueta y 6 tipos de goma.
- El montaje experimental restringe el movimiento a un plano 2D.
- La máquina lanzapelotas limita el rango de velocidad y spin.
- Las propiedades de la goma se describen por su tipo comercial; no se midieron
  dureza Shore, adherencia ni elasticidad de forma independiente.
- El conjunto de test del proyecto reserva R02 y R05; no representa a todas las
  raquetas del mercado.

## Citación sugerida

```bibtex
@misc{gossard2026learning,
  title={Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers for Robotic Table Tennis},
  author={Thomas Gossard},
  year={2026},
  eprint={2604.11349},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2604.11349}
}
```

El código y los datos originales son de Thomas Gossard y se distribuyen bajo
licencia MIT. Este proyecto solo agrega un pipeline de análisis y modelos.
