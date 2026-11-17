# Guía rápida para GitHub y Google Colab

## 1. Subir el proyecto a GitHub

La carpeta que debe subirse es `proyecto_corregido/`. Ya incluye:

- el dataset real unificado (`datos/dataset_rebotes_tt.csv`);
- los 10 CSV originales (`datos/original/`), para trazabilidad;
- la licencia MIT del dataset (`datos/LICENSE_TT_RACKETS.txt`);
- el script de descarga (`descargar_dataset_real.py`);
- el script principal, el notebook y los gráficos/modelos generados.

Desde la carpeta `proyecto_corregido/`:

```bash
git init
git add .
git commit -m "Proyecto ML con dataset real de rebotes de tenis de mesa"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

Antes de subir, revisa que no se cuelen entornos virtuales, cachés o archivos
locales de VS Code. El archivo `.gitignore` ya los excluye.

## 2. Abrir el notebook en Google Colab

1. Sube el repositorio a GitHub.
2. En Colab, elige **Archivo > Abrir notebook > GitHub**.
3. Pega la URL del repositorio y selecciona
   `proyecto_corregido/proyecto_ml_tenis_mesa_corregido.ipynb`.
4. Ejecuta la primera celda. En Colab instalará las dependencias con
   `requirements.txt`.
5. Ejecuta el resto de celdas con **Entorno de ejecución > Ejecutar todas**.

El dataset ya está incluido, así que no necesitas descargarlo de nuevo. Si el
CSV no estuviera en el repositorio, ejecuta antes:

```python
!python descargar_dataset_real.py
```

## 3. Tiempo y recursos

- No se necesita GPU.
- El entrenamiento completo puede tardar varios minutos porque compara
  11 clasificadores y 11 regresores con `GridSearchCV`.
- Los resultados se guardan en `graficos/` y `modelos/`.
- Si Colab interrumpe la ejecución por tiempo, puedes reducir las grillas de
  hiperparámetros o ejecutar el script por partes. No cambies la separación
  train/test ni el diagnóstico de fuga.

## 4. Reproducir desde la terminal

```bash
python descargar_dataset_real.py
python proyecto_ml_tenis_mesa_corregido.py
```

En Windows:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:MPLBACKEND='Agg'
python descargar_dataset_real.py
python proyecto_ml_tenis_mesa_corregido.py
```

## 5. Qué debe decir la entrega sobre los datos

- Los datos son **reales y medidos experimentalmente** por Gossard (2026).
- La fuente es el repositorio `gossardt/tt_rackets`, con licencia MIT.
- El proyecto no usa el dataset sintético anterior para los resultados finales.
- El dataset sintético y su generador se conservan en `legacy_sintetico/` como
  evidencia del trabajo previo, pero están claramente separados.
- Las limitaciones del dataset (10 raquetas, movimiento 2D, rango limitado de
  velocidad/spin) deben mencionarse en la presentación.
