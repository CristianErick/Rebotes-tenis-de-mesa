# Proyecto ML - Rebotes de tenis de mesa con datos reales

Este proyecto compara **11 clasificadores** y **11 regresores** sobre un
dataset **real y medido experimentalmente** de rebotes de pelota de tenis de
mesa contra 10 configuraciones de raqueta.

El objetivo es doble:

1. **Clasificación:** identificar si la goma de la raqueta es **invertida** o
   **no invertida** (picos o antispin) a partir de la respuesta dinámica medida
   del rebote.
2. **Regresión:** predecir la **velocidad normal de salida** de la pelota a
   partir del estado de entrada y de la configuración de la raqueta.

El flujo conserva la metodología corregida: separación por grupos, diagnóstico
de fuga de información, validación cruzada agrupada, línea base, ajuste de
hiperparámetros y exportación ONNX.

---

## 1. Dataset real

### Fuente y licencia

- **Artículo:** Thomas Gossard. *Learning Racket-Ball Bounce Dynamics Across
  Diverse Rubbers for Robotic Table Tennis.* arXiv:2604.11349, 2026.
- **Repositorio:** https://github.com/gossardt/tt_rackets
- **Licencia:** MIT (ver `datos/LICENSE_TT_RACKETS.txt`).
- **Registros:** 8.184 rebotes reales.
- **Configuraciones:** 10 raquetas, 6 tipos de goma.
- **Variables medidas:** velocidad y spin antes y después del rebote, ángulo de
  raqueta y dirección de la trayectoria.

La metadata de cada raqueta (hoja, espesor de esponja, goma y tipo de goma) se
tomó de la Tabla IV del artículo y del README del repositorio. El detalle está
en `datos/FUENTE.md`.

### Archivos de datos

```
datos/
├── dataset_rebotes_tt.csv      # dataset unificado (8.184 x 38)
├── resumen_dataset.json        # trazabilidad y conteos
├── FUENTE.md                   # fuente, licencia y limitaciones
├── LICENSE_TT_RACKETS.txt      # licencia MIT del dataset original
└── original/
    ├── racket_01.csv
    ├── ...
    └── racket_10.csv
```

`descargar_dataset_real.py` vuelve a descargar los 10 CSV originales, agrega la
metadata y calcula las variables derivadas. No se inventan ni se imputan
registros.

### Variables principales

| Variable | Descripción |
|---|---|
| `v_in_norm` | Velocidad normal de entrada, `abs(v_in_z)` (m/s). |
| `v_in_tan` | Velocidad tangencial de entrada, `v_in_x` (m/s). |
| `spin_in` | Spin principal de entrada, `spin_before_y` (rad/s). |
| `v_out_norm` | Velocidad normal de salida, `v_out_z` (m/s). |
| `spin_out` | Spin principal de salida, `spin_after_y` (rad/s). |
| `cor` | Coeficiente de restitución, `v_out_norm / v_in_norm`. |
| `energy_ratio` | Relación de energía cinética de salida/entrada. |
| `delta_spin` | Cambio de spin, `spin_out - spin_in`. |
| `sponge_mm` | Espesor de esponja (mm). |
| `rigidez_hoja` | Código ordinal: Defensiva=0, Allrounder=1, Ofensiva=2. |
| `familia_goma` | Código ordinal: Antispin=0, Picos=1, Invertida=2. |

---

## 2. Dataset sintético anterior

El proyecto anterior usaba un dataset sintético generado por
`generar_dataset_692.py`. Ese dataset **no se usa para los resultados finales**.
Para mantener la transparencia, se conserva en `legacy_sintetico/` junto con el
generador, el notebook y los resultados originales. La carpeta está claramente
separada y no participa del pipeline actual.

---

## 3. Estructura del proyecto

```
proyecto_corregido/
├── README.md
├── GUIA_COLAB.md
├── requirements.txt
├── .gitignore
├── descargar_dataset_real.py
├── generar_notebook.py
├── resumir_resultados.py
├── proyecto_ml_tenis_mesa_corregido.py
├── proyecto_ml_tenis_mesa_corregido.ipynb
├── datos/
├── graficos/
├── modelos/
└── legacy_sintetico/
```

---

## 4. Requisitos e instalación

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `matplotlib`, `seaborn`,
`scikit-learn`, `joblib`, `scipy`, `skl2onnx`, `onnxruntime` y `nbformat`.

Probado con Python 3.13, pandas 3.0.3, numpy 2.5.1 y scikit-learn 1.9.1.

---

## 5. Cómo ejecutar

### Descargar o verificar los datos

```bash
python descargar_dataset_real.py
```

### Script principal

```bash
python proyecto_ml_tenis_mesa_corregido.py
```

En Windows:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:MPLBACKEND='Agg'
python descargar_dataset_real.py
python proyecto_ml_tenis_mesa_corregido.py
```

### Notebook

Abrir `proyecto_ml_tenis_mesa_corregido.ipynb` con Jupyter o Google Colab.
El notebook se puede regenerar desde el script con:

```bash
python generar_notebook.py
```

### Google Colab

Ver `GUIA_COLAB.md`. El dataset ya está incluido en el repositorio; no se
necesita GPU.

---

## 6. Metodología

### 6.1 Separación por configuración de raqueta

El test se reserva para **configuraciones completas no vistas**:

- **R02:** hoja ofensiva, esponja 1.8 mm, goma invertida allround.
- **R05:** hoja allrounder, esponja 1.2 mm, picos largos.

Train: configuraciones R01, R03, R04, R06, R07, R08, R09 y R10.
Test: R02 y R05.

Así se evalúa la generalización a raquetas nuevas, no a rebotes nuevos de una
raqueta ya vista. La validación cruzada se hace solo con las 8 configuraciones
de train (`StratifiedGroupKFold` para clasificación y `GroupKFold` para
regresión, 4 folds).

### 6.2 Clasificación

- **Target:** `familia_bin` (0 = No invertida, 1 = Invertida).
- **Features (10):** `v_in_norm`, `v_in_tan`, `spin_in`, `racket_angle`,
  `v_out_norm`, `v_out_tan`, `spin_out`, `cor`, `energy_ratio`, `delta_spin`.
- **Qué significa:** el modelo usa la respuesta dinámica completa del rebote
  para inferir la familia de goma. No es un predictor previo al golpe.
- **Prevención de fuga:** se excluyen `familia_goma`, `rubber_type`,
  `rubber_family` y `racket_id`, porque codifican directamente el target.
- **Validación:** `StratifiedGroupKFold(n_splits=4, shuffle=True,
  random_state=42)`.
- **Selección:** mayor F1 macro en validación cruzada.
- **Modelos:** Logistic Regression, KNN, SVM, Decision Tree, Random Forest,
  Gradient Boosting, AdaBoost, MLP, Gaussian Naive Bayes, LDA y QDA.

### 6.3 Regresión

- **Target:** `v_out_norm` (velocidad normal de salida, m/s).
- **Features (10):** `v_in_norm`, `v_in_tan`, `spin_in`, `racket_angle`,
  `sponge_mm`, `rigidez_hoja`, `familia_goma`, `angulo_incidencia`,
  `spin_relativo`, `ratio_tan_norm`.
- **Qué significa:** se predice la velocidad de salida con el estado de entrada
  y la metadata de la raqueta, sin usar ninguna variable posterior al rebote.
- **Prevención de fuga:** se excluyen `v_out_norm`, `v_out_tan`, `spin_out`,
  `cor`, `energy_ratio` y `delta_spin`.
- **Validación:** `GroupKFold(n_splits=4)`.
- **Selección:** mayor R2 en validación cruzada.
- **Modelos:** Linear Regression, Ridge, Lasso, ElasticNet, KNN, SVR, Decision
  Tree, Random Forest, Gradient Boosting, AdaBoost y MLP.

### 6.4 Línea base

- Clasificación: `DummyClassifier(strategy='most_frequent')`.
- Regresión: `DummyRegressor(strategy='mean')`.

---

## 7. Resultados

### 7.1 Diagnóstico de fuga

{{DIAGNOSTICO}}

### 7.2 Clasificación - Familia de goma

{{TABLA_CLF}}

### 7.3 Regresión - Velocidad normal de salida

{{TABLA_REG}}

### 7.4 Validación ONNX

{{ONNX}}

---

## 8. Limitaciones

- El dataset cubre solo 10 configuraciones de raqueta y 6 tipos de goma.
- El movimiento se restringe a un plano 2D; no se modela la dinámica 3D.
- La máquina lanzapelotas limita el rango de velocidad y spin.
- El test contiene solo R02 y R05; los resultados no se extrapolan a todas las
  raquetas del mercado.
- La clasificación usa la respuesta posterior al rebote; no pretende predecir
  la familia de goma antes de golpear.
- La regresión usa metadata ordinal de la raqueta, no mediciones directas de
  dureza Shore, adherencia o elasticidad.

---

## 9. Reproducibilidad

- Semilla global: `random_state=42`.
- Split de test fijo por `racket_id` (`[2, 5]`).
- Grillas de hiperparámetros declaradas en el script.
- El test no participa en la selección de modelos.
- Resultados exportados en `modelos/resultados_clasificacion.csv`,
  `modelos/resultados_regresion.csv` y `modelos/metadata.json`.
- Gráficos en `graficos/` (12 figuras).

---

## 10. Citación

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

El dataset original es de Thomas Gossard y se distribuye bajo licencia MIT.
Este repositorio agrega el pipeline de análisis, los modelos y la documentación
del proyecto académico.
