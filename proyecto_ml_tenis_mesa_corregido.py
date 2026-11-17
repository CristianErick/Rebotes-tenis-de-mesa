"""
Proyecto ML - Rebotes de tenis de mesa con datos reales.

Dataset:
    Gossard, T. (2026). Learning Racket-Ball Bounce Dynamics Across Diverse
    Rubbers for Robotic Table Tennis. arXiv:2604.11349.
    Repositorio: https://github.com/gossardt/tt_rackets (licencia MIT).

Tareas:
    1. Clasificación: identificar si la goma es invertida o no invertida
       (picos/antispin) a partir de la respuesta dinámica medida del rebote.
    2. Regresión: predecir la velocidad normal de salida de la pelota a partir
       del estado de entrada y de la configuración de la raqueta.

El flujo conserva la metodología corregida: separación por grupos, diagnóstico
de fuga, validación cruzada agrupada, línea base y exportación ONNX.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, StratifiedGroupKFold, cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

# Permite ejecutar desde la carpeta del proyecto o desde la carpeta padre.
for candidato in (BASE_DIR, BASE_DIR / 'proyecto_corregido'):
    if (candidato / 'datos' / 'dataset_rebotes_tt.csv').exists():
        BASE_DIR = candidato
        break

RUTA_DATASET = BASE_DIR / 'datos' / 'dataset_rebotes_tt.csv'
RUTA_GRAFICOS = BASE_DIR / 'graficos'
RUTA_MODELOS = BASE_DIR / 'modelos'
RUTA_GRAFICOS.mkdir(exist_ok=True)
RUTA_MODELOS.mkdir(exist_ok=True)

if not RUTA_DATASET.exists():
    raise FileNotFoundError(
        f'No se encontró el dataset real en {RUTA_DATASET}.\n'
        'Ejecuta primero: python descargar_dataset_real.py'
    )

RANDOM_STATE = 42
CV_FOLDS = 4
DPI = 300
TEST_GROUPS = [2, 5]  # R02: invertida allround; R05: picos largos.

# UTF-8 para consolas Windows.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except Exception:
    pass

plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
})
sns.set_style('whitegrid')
sns.set_palette('viridis')

print('=' * 80)
print('  PROYECTO ML - REBOTES DE TENIS DE MESA CON DATOS REALES')
print('  11 clasificadores + 11 regresores con validación por grupos')
print('=' * 80)

# %% [markdown]
# ## 1. Carga de datos y análisis exploratorio

# %%
df = pd.read_csv(RUTA_DATASET)
df['familia_bin'] = (df['rubber_family'] == 'Invertida').astype(int)

print(f'\nDataset cargado: {df.shape[0]} registros x {df.shape[1]} columnas')
print('\n--- Primeras 5 filas (columnas seleccionadas) ---')
columnas_vista = [
    'racket_id', 'rubber_type', 'sponge_mm', 'racket_angle',
    'v_in_norm', 'v_in_tan', 'spin_in', 'v_out_norm', 'spin_out', 'cor',
]
print(df[columnas_vista].head().to_string())
print('\n--- Tipos de datos ---')
print(df[columnas_vista].dtypes.to_string())
print('\n--- Valores nulos en variables de trabajo ---')
variables_trabajo = [
    'v_in_norm', 'v_in_tan', 'spin_in', 'racket_angle', 'sponge_mm',
    'rigidez_hoja', 'familia_goma', 'angulo_incidencia', 'spin_relativo',
    'ratio_tan_norm', 'v_out_norm', 'v_out_tan', 'spin_out', 'cor',
    'energy_ratio', 'delta_spin', 'familia_bin',
]
print(df[variables_trabajo].isnull().sum().to_string())
print(f"Total nulos: {df[variables_trabajo].isnull().sum().sum()}")
print(f"\nConfiguraciones de raqueta: {df['racket_id'].nunique()}")
print(f"Familias de goma: {sorted(df['rubber_family'].unique().tolist())}")
print(f"Registros por configuración:")
print(df['configuracion'].value_counts().sort_index().to_string())

vars_eda = ['v_in_norm', 'v_in_tan', 'spin_in', 'v_out_norm', 'cor']
etiquetas_vars = {
    'v_in_norm': 'Velocidad normal de entrada (m/s)',
    'v_in_tan': 'Velocidad tangencial de entrada (m/s)',
    'spin_in': 'Spin de entrada (rad/s)',
    'v_out_norm': 'Velocidad normal de salida (m/s)',
    'cor': 'Coeficiente de restitución (COR)',
}
print('\n--- Estadísticas descriptivas ---')
print(df[vars_eda].describe().round(4).to_string())

# %% [markdown]
# ### Gráficos 01-03: EDA

# %%
print('\nGenerando gráfico 01: Distribuciones con KDE...')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
colores = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
for i, (var, color) in enumerate(zip(vars_eda, colores)):
    ax = axes[i]
    sns.histplot(df[var], kde=True, bins=30, color=color, ax=ax,
                 edgecolor='white', alpha=0.7)
    ax.set_title(f'Distribución de {etiquetas_vars[var]}', fontweight='bold')
    ax.set_xlabel(etiquetas_vars[var])
    ax.set_ylabel('Frecuencia')
    media = df[var].mean()
    mediana = df[var].median()
    ax.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'Media: {media:.3f}')
    ax.axvline(mediana, color='black', linestyle='-.', linewidth=1.5, label=f'Mediana: {mediana:.3f}')
    ax.legend(fontsize=8)
axes[5].set_visible(False)
fig.suptitle('Distribución de Variables Físicas del Rebote - Dataset Real',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '01_eda_distribuciones.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 02: Mapa de calor de correlación...')
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[vars_eda].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, mask=mask,
            square=True, linewidths=2, linecolor='white',
            cbar_kws={'label': 'Coeficiente de Pearson', 'shrink': 0.8},
            annot_kws={'size': 12, 'fontweight': 'bold'}, ax=ax)
ax.set_title('Matriz de Correlación de Pearson - Variables del Rebote',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels([etiquetas_vars[v] for v in vars_eda], rotation=35, ha='right')
ax.set_yticklabels([etiquetas_vars[v] for v in vars_eda], rotation=0)
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '02_eda_correlacion.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 03: Boxplots de variables físicas...')
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
for i, (var, color) in enumerate(zip(vars_eda, colores)):
    axes[i].boxplot(df[var].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    axes[i].set_title(etiquetas_vars[var], fontweight='bold', fontsize=10)
    axes[i].set_ylabel('Valor')
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[var] < (q1 - 1.5 * iqr)) | (df[var] > (q3 + 1.5 * iqr))).sum()
    axes[i].text(0.5, -0.15, f'Outliers: {outliers}', transform=axes[i].transAxes,
                 ha='center', fontsize=9, color='red', fontweight='bold')
fig.suptitle('Diagramas de Caja - Detección de Valores Atípicos', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '03_eda_boxplots.png', dpi=DPI, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 2. Separación por configuración de raqueta y diagnóstico de fuga
#
# El test se reserva para configuraciones de raqueta completas no vistas:
# R02 (invertida allround) y R05 (picos largos). Así se evalúa la capacidad
# de generalizar a raquetas nuevas, no a rebotes nuevos de la misma raqueta.
#
# La clasificación usa la respuesta dinámica medida (antes y después del
# rebote) para identificar la familia de goma. No usa la metadata que codifica
# directamente el target (`familia_goma`, `rubber_type` ni `racket_id`).
# La regresión usa solo el estado de entrada y la metadata de la raqueta para
# predecir la velocidad normal de salida.

# %%
df_train = df[~df['racket_id'].isin(TEST_GROUPS)].copy()
df_test = df[df['racket_id'].isin(TEST_GROUPS)].copy()

groups_train = df_train['racket_id'].copy()
groups_test = df_test['racket_id'].copy()

print(f'\nConfiguraciones de test reservadas: {TEST_GROUPS}')
print(f'Configuraciones de train: {sorted(df_train["racket_id"].unique().tolist())}')
print(f'Train: {len(df_train)} rebotes ({groups_train.nunique()} configuraciones)')
print(f'Test:  {len(df_test)} rebotes ({groups_test.nunique()} configuraciones)')

# --- Diagnóstico de fuga en clasificación ---
features_clf = [
    'v_in_norm', 'v_in_tan', 'spin_in', 'racket_angle',
    'v_out_norm', 'v_out_tan', 'spin_out', 'cor',
    'energy_ratio', 'delta_spin',
]
features_reg = [
    'v_in_norm', 'v_in_tan', 'spin_in', 'racket_angle',
    'sponge_mm', 'rigidez_hoja', 'familia_goma',
    'angulo_incidencia', 'spin_relativo', 'ratio_tan_norm',
]
features_raw_clf = [
    'v_in_norm', 'v_in_tan', 'spin_in', 'racket_angle',
    'v_out_norm', 'v_out_tan', 'spin_out',
]

X_train_clf = df_train[features_clf].copy()
X_test_clf = df_test[features_clf].copy()
y_train_clf = df_train['familia_bin'].copy()
y_test_clf = df_test['familia_bin'].copy()

X_train_reg = df_train[features_reg].copy()
X_test_reg = df_test[features_reg].copy()
y_train_reg = df_train['v_out_norm'].copy()
y_test_reg = df_test['v_out_norm'].copy()

etiquetas_clf = {0: 'No invertida', 1: 'Invertida'}

print('\n--- Diagnóstico de fuga en clasificación ---')
diag_clf_leak = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])
diag_clf_no = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])
diag_clf_leak.fit(df_train[features_clf + ['familia_goma']], y_train_clf)
diag_clf_no.fit(df_train[features_clf], y_train_clf)
acc_leak = accuracy_score(
    y_test_clf, diag_clf_leak.predict(df_test[features_clf + ['familia_goma']])
)
acc_no = accuracy_score(y_test_clf, diag_clf_no.predict(df_test[features_clf]))
print(f'Accuracy con metadata que codifica el target (fuga): {acc_leak:.4f}')
print(f'Accuracy sin esa metadata (modelo principal):         {acc_no:.4f}')
print('La diferencia confirma que la metadata de la goma no debe usarse como feature de clasificación.')

print('\n--- Diagnóstico de fuga en regresión ---')
diag_reg_leak = LinearRegression()
diag_reg_no = LinearRegression()
diag_reg_leak.fit(df_train[features_reg + ['cor']], y_train_reg)
diag_reg_no.fit(df_train[features_reg], y_train_reg)
r2_leak = r2_score(y_test_reg, diag_reg_leak.predict(df_test[features_reg + ['cor']]))
r2_no = r2_score(y_test_reg, diag_reg_no.predict(df_test[features_reg]))
print(f'R2 con COR (reconstruye el target, fuga): {r2_leak:.4f}')
print(f'R2 sin variables de salida (modelo principal): {r2_no:.4f}')

print('\n--- Distribución del target de clasificación ---')
for clase, nombre in etiquetas_clf.items():
    n_train = int((y_train_clf == clase).sum())
    n_test = int((y_test_clf == clase).sum())
    print(f'  {nombre:14s}: train {n_train:4d} | test {n_test:4d}')

print(f'\nFeatures de clasificación ({len(features_clf)}): {features_clf}')
print(f'Features de regresión ({len(features_reg)}): {features_reg}')

# %% [markdown]
# ## 3. Clasificación: 11 modelos con Pipelines y GridSearchCV
#
# Target: `familia_bin` (Invertida vs No invertida).
# La selección se hace por F1 macro en validación cruzada agrupada.
# El test se reserva para la evaluación final.

# %%
cv_clf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_reg = GroupKFold(n_splits=CV_FOLDS)

SIN_ESCALADO_CLF = {'Decision Tree', 'Random Forest', 'Gradient Boosting',
                    'AdaBoost', 'Gaussian Naive Bayes'}
SIN_ESCALADO_REG = {'Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost'}


def crear_pipeline(modelo, escalar):
    pasos = []
    if escalar:
        pasos.append(('scaler', StandardScaler()))
    pasos.append(('model', modelo))
    return Pipeline(pasos)


clasificadores = {
    'Logistic Regression': (
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        {'model__C': [0.01, 0.1, 1.0, 10.0]},
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {'model__n_neighbors': [3, 5, 7, 11], 'model__weights': ['uniform', 'distance']},
    ),
    'SVM (SVC)': (
        SVC(probability=True, random_state=RANDOM_STATE),
        {'model__C': [0.1, 1.0, 10.0], 'model__gamma': ['scale']},
    ),
    'Decision Tree': (
        DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE),
        {'model__max_depth': [3, 5, 7, None], 'model__min_samples_leaf': [1, 3]},
    ),
    'Random Forest': (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        {'model__max_depth': [None, 8, 12], 'model__min_samples_leaf': [1, 3]},
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1],
         'model__max_depth': [2, 3]},
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.5, 1.0]},
    ),
    'MLP Neural Network': (
        MLPClassifier(max_iter=3000, random_state=RANDOM_STATE, early_stopping=True),
        {'model__hidden_layer_sizes': [(50,), (100,), (50, 25)],
         'model__alpha': [0.0001, 0.001]},
    ),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {'model__solver': ['svd', 'lsqr']} ),
    'QDA': (QuadraticDiscriminantAnalysis(), {'model__reg_param': [0.01, 0.1, 0.3]}),
}

print('\n--- Entrenamiento y evaluación de 11 clasificadores ---')
resultados_clf = []
modelos_clf_entrenados = {}
predicciones_clf = {}

for nombre, (modelo, grid) in clasificadores.items():
    pipe = crear_pipeline(modelo, nombre not in SIN_ESCALADO_CLF)
    print(f'   ... {nombre}')
    try:
        gs = GridSearchCV(
            pipe, grid, cv=cv_clf, scoring='f1_macro',
            n_jobs=-1, refit=True, error_score='raise',
        )
        gs.fit(X_train_clf, y_train_clf, groups=groups_train)
        best = gs.best_estimator_

        cv_res = cross_validate(
            best, X_train_clf, y_train_clf, groups=groups_train, cv=cv_clf,
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            n_jobs=1,
        )
        y_pred = best.predict(X_test_clf)

        acc = accuracy_score(y_test_clf, y_pred)
        prec = precision_score(y_test_clf, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test_clf, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, average='macro', zero_division=0)

        resultados_clf.append({
            'Tipo': 'Modelo',
            'Modelo': nombre,
            'F1_CV': cv_res['test_f1_macro'].mean(),
            'F1_CV_std': cv_res['test_f1_macro'].std(),
            'Accuracy_CV': cv_res['test_accuracy'].mean(),
            'Accuracy_CV_std': cv_res['test_accuracy'].std(),
            'Accuracy_Test': acc,
            'Precision_Test': prec,
            'Recall_Test': rec,
            'F1_Test': f1,
            'Mejores_Parametros': str(gs.best_params_),
        })
        modelos_clf_entrenados[nombre] = best
        predicciones_clf[nombre] = y_pred
        print(f'   {nombre:22s} -> F1 CV: {cv_res["test_f1_macro"].mean():.4f} '
              f'+/- {cv_res["test_f1_macro"].std():.4f} | F1 Test: {f1:.4f}')
    except Exception as exc:
        print(f'   {nombre}: falló ({exc})')
        resultados_clf.append({
            'Tipo': 'Modelo', 'Modelo': nombre,
            'F1_CV': np.nan, 'F1_CV_std': np.nan,
            'Accuracy_CV': np.nan, 'Accuracy_CV_std': np.nan,
            'Accuracy_Test': np.nan, 'Precision_Test': np.nan,
            'Recall_Test': np.nan, 'F1_Test': np.nan,
            'Mejores_Parametros': 'ERROR',
        })

baseline_clf = DummyClassifier(strategy='most_frequent')
cv_base_clf = cross_validate(
    baseline_clf, X_train_clf, y_train_clf, groups=groups_train, cv=cv_clf,
    scoring=['accuracy', 'f1_macro'], n_jobs=1,
)
baseline_clf.fit(X_train_clf, y_train_clf)
y_pred_base_clf = baseline_clf.predict(X_test_clf)
resultados_clf.append({
    'Tipo': 'Baseline',
    'Modelo': 'Baseline (clase mayoritaria)',
    'F1_CV': cv_base_clf['test_f1_macro'].mean(),
    'F1_CV_std': cv_base_clf['test_f1_macro'].std(),
    'Accuracy_CV': cv_base_clf['test_accuracy'].mean(),
    'Accuracy_CV_std': cv_base_clf['test_accuracy'].std(),
    'Accuracy_Test': accuracy_score(y_test_clf, y_pred_base_clf),
    'Precision_Test': precision_score(y_test_clf, y_pred_base_clf, average='macro', zero_division=0),
    'Recall_Test': recall_score(y_test_clf, y_pred_base_clf, average='macro', zero_division=0),
    'F1_Test': f1_score(y_test_clf, y_pred_base_clf, average='macro', zero_division=0),
    'Mejores_Parametros': '-',
})

df_resultados_clf = pd.DataFrame(resultados_clf).sort_values(
    'F1_CV', ascending=False, na_position='last',
).reset_index(drop=True)
df_resultados_clf.index += 1
df_resultados_clf.to_csv(RUTA_MODELOS / 'resultados_clasificacion.csv', index=True)

print('\n' + '=' * 80)
print('  TABLA COMPARATIVA DE CLASIFICADORES (ordenada por F1 CV)')
print('=' * 80)
print(df_resultados_clf.to_string())
print('\n   Exportado: modelos/resultados_clasificacion.csv')

fila_mejor_clf = df_resultados_clf[df_resultados_clf['Tipo'] == 'Modelo'].iloc[0]
mejor_clf_nombre = fila_mejor_clf['Modelo']
mejor_clf_pipeline = modelos_clf_entrenados[mejor_clf_nombre]
mejor_clf_f1_cv = fila_mejor_clf['F1_CV']
mejor_clf_f1_test = fila_mejor_clf['F1_Test']
y_pred_mejor_clf = predicciones_clf[mejor_clf_nombre]

print(f'\nMEJOR CLASIFICADOR (por F1 CV): {mejor_clf_nombre}')
print(f'   F1 CV:   {mejor_clf_f1_cv:.4f} +/- {fila_mejor_clf["F1_CV_std"]:.4f}')
print(f'   F1 Test: {mejor_clf_f1_test:.4f}')
print(f'   Accuracy Test: {fila_mejor_clf["Accuracy_Test"]:.4f}')

# %% [markdown]
# ### Gráficos 04-09: evaluación de clasificación

# %%
print('\nGenerando gráfico 04: Comparativa de clasificadores...')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
df_plot_clf = df_resultados_clf[df_resultados_clf['Tipo'] == 'Modelo'].dropna(subset=['F1_CV'])
df_plot_clf = df_plot_clf.sort_values('F1_CV', ascending=True)
baseline_clf_f1_cv = df_resultados_clf.loc[
    df_resultados_clf['Tipo'] == 'Baseline', 'F1_CV'
].values[0]
baseline_clf_f1_test = df_resultados_clf.loc[
    df_resultados_clf['Tipo'] == 'Baseline', 'F1_Test'
].values[0]
colores_barras = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot_clf)))

bars1 = axes[0].barh(df_plot_clf['Modelo'], df_plot_clf['F1_CV'],
                     xerr=df_plot_clf['F1_CV_std'], color=colores_barras,
                     edgecolor='white', height=0.6, capsize=4)
for bar, val in zip(bars1, df_plot_clf['F1_CV']):
    axes[0].text(val * 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='center', fontweight='bold', fontsize=9,
                 color='black',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75,
                           edgecolor='none'))
axes[0].axvline(baseline_clf_f1_cv, color='red', linestyle='--', linewidth=2,
                label=f'Baseline: {baseline_clf_f1_cv:.3f}')
axes[0].set_xlabel('F1 macro (validación cruzada agrupada)')
axes[0].set_title('F1 CV por Clasificador', fontweight='bold')
axes[0].legend()

bars2 = axes[1].barh(df_plot_clf['Modelo'], df_plot_clf['F1_Test'],
                     color=colores_barras, edgecolor='white', height=0.6)
for bar, val in zip(bars2, df_plot_clf['F1_Test']):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontweight='bold', fontsize=9)
axes[1].axvline(baseline_clf_f1_test, color='red', linestyle='--', linewidth=2,
                label=f'Baseline: {baseline_clf_f1_test:.3f}')
axes[1].set_xlabel('F1 macro (test)')
axes[1].set_title('F1 Test por Clasificador', fontweight='bold')
axes[1].legend()
fig.suptitle('Comparativa de 11 Modelos de Clasificación - Familia de Goma',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '04_clf_comparativa_modelos.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 05: Matriz de confusión del mejor clasificador...')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm = confusion_matrix(y_test_clf, y_pred_mejor_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No invertida', 'Invertida'])
disp.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].grid(False)
axes[0].set_title(f'Matriz de Confusión\n{mejor_clf_nombre}', fontweight='bold')
report = classification_report(
    y_test_clf, y_pred_mejor_clf,
    target_names=['No invertida', 'Invertida'],
)
axes[1].text(0.1, 0.5, report, transform=axes[1].transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[1].set_title(f'Reporte de Clasificación\n{mejor_clf_nombre}', fontweight='bold')
axes[1].axis('off')
print(f'\n--- Reporte de Clasificación: {mejor_clf_nombre} ---')
print(report)
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '05_clf_matriz_confusion_mejor.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 06: Cuadrícula de matrices de confusión (3x4)...')
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes_flat = axes.flatten()
nombres_clf = list(clasificadores.keys())
for i, nombre in enumerate(nombres_clf):
    if nombre not in predicciones_clf:
        axes_flat[i].set_visible(False)
        continue
    cm_i = confusion_matrix(y_test_clf, predicciones_clf[nombre])
    disp_i = ConfusionMatrixDisplay(confusion_matrix=cm_i,
                                    display_labels=['No inv.', 'Invertida'])
    disp_i.plot(ax=axes_flat[i], cmap='YlOrRd', values_format='d', colorbar=False)
    axes_flat[i].grid(False)
    fila = df_resultados_clf[df_resultados_clf['Modelo'] == nombre].iloc[0]
    axes_flat[i].set_title(
        f'{nombre}\nCV: {fila["F1_CV"]:.2f} | Test: {fila["F1_Test"]:.2f}',
        fontsize=9, fontweight='bold',
    )
    axes_flat[i].set_xlabel('')
    axes_flat[i].set_ylabel('')
axes_flat[11].set_visible(False)
fig.suptitle('Matrices de Confusión - 11 Clasificadores (test)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '06_clf_matrices_confusion_grid.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 07: Árbol de Decisión...')
fig, ax = plt.subplots(figsize=(24, 12))
dt_pipeline = modelos_clf_entrenados['Decision Tree']
dt_model = dt_pipeline.named_steps['model']
plot_tree(dt_model, feature_names=features_clf,
          class_names=['No invertida', 'Invertida'],
          filled=True, rounded=True, ax=ax, fontsize=7, max_depth=4,
          proportion=True, impurity=True)
ax.set_title('Árbol de Decisión - Familia de Goma (profundidad visualizada: 4)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '07_clf_arbol_decision.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 08: Importancia de variables...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, nombre_modelo in zip(axes, ['Random Forest', 'Gradient Boosting']):
    pipeline = modelos_clf_entrenados[nombre_modelo]
    modelo = pipeline.named_steps['model']
    importancia = pd.Series(modelo.feature_importances_, index=features_clf).sort_values(ascending=True)
    colores_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importancia)))
    ax.barh(importancia.index, importancia.values, color=colores_imp, edgecolor='white')
    for i, val in enumerate(importancia.values):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax.set_title(f'Importancia de Variables\n({nombre_modelo})', fontweight='bold')
    ax.set_xlabel('Importancia')
fig.suptitle('Importancia de Características - Clasificación de Familia de Goma',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '08_clf_importancia_variables.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 09: Fronteras de decisión 2D...')
rf_pipeline = modelos_clf_entrenados['Random Forest']
rf_model = rf_pipeline.named_steps['model']
importancia_total = pd.Series(rf_model.feature_importances_, index=features_clf).sort_values(ascending=False)
importancia_raw = importancia_total[features_raw_clf].sort_values(ascending=False)
top2_features = importancia_raw.index[:2].tolist()
print(f'   Variables RAW más influyentes: {top2_features[0]}, {top2_features[1]}')

X_2d_train = X_train_clf[top2_features].copy()
X_2d_test = X_test_clf[top2_features].copy()

modelos_2d = {
    'Logistic Regression': crear_pipeline(
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), True),
    'K-Nearest Neighbors': crear_pipeline(KNeighborsClassifier(), True),
    'SVM (SVC)': crear_pipeline(
        SVC(probability=True, random_state=RANDOM_STATE), True),
    'Random Forest': crear_pipeline(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), False),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()
colores_clases = ['#FF6B6B', '#4ECDC4']
nombres_clases = ['No invertida', 'Invertida']
for i, (nombre_m, modelo_m) in enumerate(modelos_2d.items()):
    ax = axes_flat[i]
    modelo_m.fit(X_2d_train, y_train_clf)
    x_min, x_max = X_2d_train.iloc[:, 0].min(), X_2d_train.iloc[:, 0].max()
    y_min, y_max = X_2d_train.iloc[:, 1].min(), X_2d_train.iloc[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    h = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min - x_pad, x_max + x_pad, h),
        np.arange(y_min - y_pad, y_max + y_pad, h),
    )
    malla = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=top2_features)
    Z = modelo_m.predict(malla).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
    for clase in [0, 1]:
        mask = (y_test_clf == clase).values
        ax.scatter(X_2d_test.iloc[mask, 0], X_2d_test.iloc[mask, 1],
                   c=colores_clases[clase], label=nombres_clases[clase],
                   edgecolors='black', s=30, alpha=0.7)
    acc_2d = accuracy_score(y_test_clf, modelo_m.predict(X_2d_test))
    ax.set_title(f'{nombre_m}\nAcc 2D Test: {acc_2d:.2%}', fontweight='bold')
    ax.set_xlabel(top2_features[0].replace('_', ' ').title())
    ax.set_ylabel(top2_features[1].replace('_', ' ').title())
    ax.legend(fontsize=8, loc='best')
fig.suptitle(f'Fronteras de Decisión 2D ({top2_features[0]} vs {top2_features[1]})',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '09_clf_fronteras_decision_2d.png', dpi=DPI, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 4. Regresión: 11 modelos con Pipelines y GridSearchCV
#
# Target: `v_out_norm` (velocidad normal de salida, m/s).
# Features: estado de entrada + metadata de la raqueta. No se usan variables
# posteriores al rebote. Selección por R2 en validación cruzada agrupada.

# %%
regresores = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge': (
        Ridge(random_state=RANDOM_STATE),
        {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    'Lasso': (
        Lasso(random_state=RANDOM_STATE, max_iter=20000),
        {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]},
    ),
    'ElasticNet': (
        ElasticNet(random_state=RANDOM_STATE, max_iter=20000),
        {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
         'model__l1_ratio': [0.2, 0.5, 0.8]},
    ),
    'K-Nearest Neighbors': (
        KNeighborsRegressor(),
        {'model__n_neighbors': [3, 5, 7, 9, 11], 'model__weights': ['uniform']},
    ),
    'SVR': (
        SVR(),
        {'model__C': [0.1, 1.0, 10.0], 'model__gamma': ['scale'],
         'model__epsilon': [0.01, 0.1]},
    ),
    'Decision Tree': (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {'model__max_depth': [3, 5, 7, None], 'model__min_samples_leaf': [1, 3, 5]},
    ),
    'Random Forest': (
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        {'model__max_depth': [None, 8, 12], 'model__min_samples_leaf': [1, 3]},
    ),
    'Gradient Boosting': (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1],
         'model__max_depth': [2, 3]},
    ),
    'AdaBoost': (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.5, 1.0]},
    ),
    'MLP Neural Network': (
        MLPRegressor(max_iter=3000, random_state=RANDOM_STATE, early_stopping=True),
        {'model__hidden_layer_sizes': [(50,), (100,), (50, 25)],
         'model__alpha': [0.0001, 0.001]},
    ),
}

print('\n--- Entrenamiento y evaluación de 11 regresores ---')
resultados_reg = []
modelos_reg_entrenados = {}
predicciones_reg = {}

for nombre, (modelo, grid) in regresores.items():
    pipe = crear_pipeline(modelo, nombre not in SIN_ESCALADO_REG)
    print(f'   ... {nombre}')
    try:
        gs = GridSearchCV(
            pipe, grid, cv=cv_reg, scoring='neg_root_mean_squared_error',
            n_jobs=-1, refit=True, error_score='raise',
        )
        gs.fit(X_train_reg, y_train_reg, groups=groups_train)
        best = gs.best_estimator_

        cv_res = cross_validate(
            best, X_train_reg, y_train_reg, groups=groups_train, cv=cv_reg,
            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            n_jobs=1,
        )
        y_pred = best.predict(X_test_reg)

        r2 = r2_score(y_test_reg, y_pred)
        mae = mean_absolute_error(y_test_reg, y_pred)
        mse = mean_squared_error(y_test_reg, y_pred)
        rmse = np.sqrt(mse)

        resultados_reg.append({
            'Tipo': 'Modelo',
            'Modelo': nombre,
            'R2_CV': cv_res['test_r2'].mean(),
            'R2_CV_std': cv_res['test_r2'].std(),
            'RMSE_CV': -cv_res['test_neg_root_mean_squared_error'].mean(),
            'RMSE_CV_std': cv_res['test_neg_root_mean_squared_error'].std(),
            'MAE_CV': -cv_res['test_neg_mean_absolute_error'].mean(),
            'R2_Test': r2,
            'MAE_Test': mae,
            'MSE_Test': mse,
            'RMSE_Test': rmse,
            'Mejores_Parametros': str(gs.best_params_),
        })
        modelos_reg_entrenados[nombre] = best
        predicciones_reg[nombre] = y_pred
        print(f'   {nombre:22s} -> R2 CV: {cv_res["test_r2"].mean():.4f} '
              f'+/- {cv_res["test_r2"].std():.4f} | RMSE Test: {rmse:.4f}')
    except Exception as exc:
        print(f'   {nombre}: falló ({exc})')
        resultados_reg.append({
            'Tipo': 'Modelo', 'Modelo': nombre,
            'R2_CV': np.nan, 'R2_CV_std': np.nan,
            'RMSE_CV': np.nan, 'RMSE_CV_std': np.nan, 'MAE_CV': np.nan,
            'R2_Test': np.nan, 'MAE_Test': np.nan, 'MSE_Test': np.nan,
            'RMSE_Test': np.nan, 'Mejores_Parametros': 'ERROR',
        })

baseline_reg = DummyRegressor(strategy='mean')
cv_base_reg = cross_validate(
    baseline_reg, X_train_reg, y_train_reg, groups=groups_train, cv=cv_reg,
    scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=1,
)
baseline_reg.fit(X_train_reg, y_train_reg)
y_pred_base_reg = baseline_reg.predict(X_test_reg)
resultados_reg.append({
    'Tipo': 'Baseline',
    'Modelo': 'Baseline (media)',
    'R2_CV': cv_base_reg['test_r2'].mean(),
    'R2_CV_std': cv_base_reg['test_r2'].std(),
    'RMSE_CV': -cv_base_reg['test_neg_root_mean_squared_error'].mean(),
    'RMSE_CV_std': cv_base_reg['test_neg_root_mean_squared_error'].std(),
    'MAE_CV': -cv_base_reg['test_neg_mean_absolute_error'].mean(),
    'R2_Test': r2_score(y_test_reg, y_pred_base_reg),
    'MAE_Test': mean_absolute_error(y_test_reg, y_pred_base_reg),
    'MSE_Test': mean_squared_error(y_test_reg, y_pred_base_reg),
    'RMSE_Test': np.sqrt(mean_squared_error(y_test_reg, y_pred_base_reg)),
    'Mejores_Parametros': '-',
})

df_resultados_reg = pd.DataFrame(resultados_reg).sort_values(
    'R2_CV', ascending=False, na_position='last',
).reset_index(drop=True)
df_resultados_reg.index += 1
df_resultados_reg.to_csv(RUTA_MODELOS / 'resultados_regresion.csv', index=True)

print('\n' + '=' * 80)
print('  TABLA COMPARATIVA DE REGRESORES (ordenada por R2 CV)')
print('=' * 80)
print(df_resultados_reg.to_string())
print('\n   Exportado: modelos/resultados_regresion.csv')

fila_mejor_reg = df_resultados_reg[df_resultados_reg['Tipo'] == 'Modelo'].iloc[0]
mejor_reg_nombre = fila_mejor_reg['Modelo']
mejor_reg_pipeline = modelos_reg_entrenados[mejor_reg_nombre]
mejor_reg_r2_cv = fila_mejor_reg['R2_CV']
mejor_reg_r2_test = fila_mejor_reg['R2_Test']
mejor_reg_rmse_test = fila_mejor_reg['RMSE_Test']
y_pred_mejor_reg = predicciones_reg[mejor_reg_nombre]

print(f'\nMEJOR REGRESOR (por R2 CV): {mejor_reg_nombre}')
print(f'   R2 CV:   {mejor_reg_r2_cv:.4f} +/- {fila_mejor_reg["R2_CV_std"]:.4f}')
print(f'   R2 Test: {mejor_reg_r2_test:.4f}')
print(f'   RMSE Test: {mejor_reg_rmse_test:.4f}')

# %% [markdown]
# ### Gráficos 10-12: evaluación de regresión

# %%
print('\nGenerando gráfico 10: Comparativa R2 y RMSE...')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
df_plot_reg = df_resultados_reg[df_resultados_reg['Tipo'] == 'Modelo'].dropna(subset=['R2_CV'])
df_plot_reg = df_plot_reg.sort_values('R2_CV', ascending=True)
baseline_reg_r2 = df_resultados_reg.loc[
    df_resultados_reg['Tipo'] == 'Baseline', 'R2_Test'
].values[0]
baseline_reg_rmse = df_resultados_reg.loc[
    df_resultados_reg['Tipo'] == 'Baseline', 'RMSE_Test'
].values[0]
colores_reg = plt.cm.plasma(np.linspace(0.2, 0.9, len(df_plot_reg)))

bars_r2 = axes[0].barh(df_plot_reg['Modelo'], df_plot_reg['R2_CV'],
                       xerr=df_plot_reg['R2_CV_std'], color=colores_reg,
                       edgecolor='white', height=0.6, capsize=4)
for bar, val in zip(bars_r2, df_plot_reg['R2_CV']):
    axes[0].text(val * 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='center', fontweight='bold', fontsize=9,
                 color='black',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75,
                           edgecolor='none'))
axes[0].scatter(df_plot_reg['R2_Test'], np.arange(len(df_plot_reg)),
                marker='D', color='black', zorder=5, label='Test')
axes[0].axvline(baseline_reg_r2, color='red', linestyle='--', linewidth=2,
                label=f'Baseline test: {baseline_reg_r2:.3f}')
axes[0].axvline(0, color='gray', linestyle=':', linewidth=1)
axes[0].set_xlabel('R2 (validación cruzada agrupada)')
axes[0].set_title('R2 CV (barras) y R2 Test (rombos)', fontweight='bold')
axes[0].legend()

df_plot_rmse = df_plot_reg.sort_values('RMSE_CV', ascending=False)
bars_rmse = axes[1].barh(df_plot_rmse['Modelo'], df_plot_rmse['RMSE_CV'],
                         xerr=df_plot_rmse['RMSE_CV_std'],
                         color=plt.cm.plasma(np.linspace(0.9, 0.2, len(df_plot_rmse))),
                         edgecolor='white', height=0.6, capsize=4)
for bar, val in zip(bars_rmse, df_plot_rmse['RMSE_CV']):
    axes[1].text(val * 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='center', fontweight='bold', fontsize=9,
                 color='black',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75,
                           edgecolor='none'))
axes[1].scatter(df_plot_rmse['RMSE_Test'], np.arange(len(df_plot_rmse)),
                marker='D', color='black', zorder=5, label='Test')
axes[1].axvline(baseline_reg_rmse, color='red', linestyle='--', linewidth=2,
                label=f'Baseline test: {baseline_reg_rmse:.3f}')
axes[1].set_xlabel('RMSE (validación cruzada agrupada)')
axes[1].set_title('RMSE CV (barras) y RMSE Test (rombos)', fontweight='bold')
axes[1].legend()
fig.suptitle('Comparativa de 11 Modelos de Regresión - Velocidad Normal de Salida',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '10_reg_comparativa_r2_rmse.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 11: Valores reales vs predichos...')
top4_reg = df_resultados_reg[df_resultados_reg['Tipo'] == 'Modelo'].head(4)['Modelo'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()
for i, nombre_m in enumerate(top4_reg):
    ax = axes_flat[i]
    y_pred_m = predicciones_reg[nombre_m]
    r2_m = r2_score(y_test_reg, y_pred_m)
    ax.scatter(y_test_reg, y_pred_m, alpha=0.5, edgecolors='black',
               linewidth=0.4, s=25, c='#2196F3')
    lim_min = min(y_test_reg.min(), y_pred_m.min()) - 0.3
    lim_max = max(y_test_reg.max(), y_pred_m.max()) + 0.3
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Ideal (y = x)')
    ax.set_xlabel('Valores Reales (v_out_norm, m/s)')
    ax.set_ylabel('Valores Predichos')
    ax.set_title(f'{nombre_m}\nR2 Test = {r2_m:.4f}', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
fig.suptitle('Valores Reales vs Predichos - Top 4 Regresores (por R2 CV)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '11_reg_real_vs_predicho.png', dpi=DPI, bbox_inches='tight')
plt.close()

print('Generando gráfico 12: Residuales e importancia de variables...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
residuales = y_test_reg.values - y_pred_mejor_reg
axes[0].hist(residuales, bins=30, color='#2196F3', edgecolor='white', alpha=0.7, density=True)
try:
    from scipy.stats import norm
    mu_res, std_res = norm.fit(residuales)
    x_range = np.linspace(residuales.min() - 0.5, residuales.max() + 0.5, 100)
    axes[0].plot(x_range, norm.pdf(x_range, mu_res, std_res), 'r-', linewidth=2,
                 label=f'Normal (mu={mu_res:.3f}, sigma={std_res:.3f})')
except Exception:
    pass
axes[0].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Error = 0')
axes[0].set_title(f'Distribución de Residuales\n{mejor_reg_nombre}', fontweight='bold')
axes[0].set_xlabel('Error Residual (Real - Predicho)')
axes[0].set_ylabel('Densidad')
axes[0].legend(fontsize=9)

modelo_imp = mejor_reg_pipeline.named_steps['model']
if hasattr(modelo_imp, 'feature_importances_'):
    importancia_reg = pd.Series(modelo_imp.feature_importances_, index=features_reg).sort_values(ascending=True)
    etiqueta_imp = f'Importancia de Variables\n({mejor_reg_nombre})'
    xlabel_imp = 'Importancia'
elif hasattr(modelo_imp, 'coef_'):
    importancia_reg = pd.Series(np.abs(np.ravel(modelo_imp.coef_)), index=features_reg).sort_values(ascending=True)
    etiqueta_imp = f'Coeficientes Absolutos\n({mejor_reg_nombre})'
    xlabel_imp = '|Coeficiente|'
else:
    importancia_reg = pd.Series(np.zeros(len(features_reg)), index=features_reg)
    etiqueta_imp = f'Sin importancia disponible\n({mejor_reg_nombre})'
    xlabel_imp = 'Importancia'
colores_imp_reg = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(importancia_reg)))
axes[1].barh(importancia_reg.index, importancia_reg.values, color=colores_imp_reg, edgecolor='white')
for j, val in enumerate(importancia_reg.values):
    axes[1].text(val + 0.005, j, f'{val:.4f}', va='center', fontweight='bold', fontsize=10)
axes[1].set_title(etiqueta_imp, fontweight='bold')
axes[1].set_xlabel(xlabel_imp)
fig.suptitle('Análisis de Residuales e Importancia - Regresión', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS / '12_reg_residuales_e_importancia.png', dpi=DPI, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 5. Serialización, ONNX e inferencia
#
# Los paquetes `.joblib` guardan el pipeline completo. Los `.onnx` también
# incluyen el escalado, por lo que aceptan directamente las features originales.

# %%
print('\n' + '=' * 80)
print('  SERIALIZACIÓN Y PRUEBA DE INFERENCIA')
print('=' * 80)

paquete_clf = {
    'pipeline': mejor_clf_pipeline,
    'features': features_clf,
    'nombre': mejor_clf_nombre,
    'metricas': {
        'f1_cv': float(mejor_clf_f1_cv),
        'f1_cv_std': float(fila_mejor_clf['F1_CV_std']),
        'f1_test': float(mejor_clf_f1_test),
        'accuracy_test': float(fila_mejor_clf['Accuracy_Test']),
    },
    'etiquetas': etiquetas_clf,
    'target': 'familia_bin (Invertida vs No invertida)',
    'grupos': 'racket_id',
    'baseline': {
        'f1_cv': float(baseline_clf_f1_cv),
        'f1_test': float(baseline_clf_f1_test),
    },
}
ruta_mejor_clf = RUTA_MODELOS / 'mejor_clasificador.joblib'
joblib.dump(paquete_clf, ruta_mejor_clf)
print(f'   Mejor clasificador guardado: {ruta_mejor_clf}')

paquete_reg = {
    'pipeline': mejor_reg_pipeline,
    'features': features_reg,
    'nombre': mejor_reg_nombre,
    'metricas': {
        'r2_cv': float(mejor_reg_r2_cv),
        'r2_cv_std': float(fila_mejor_reg['R2_CV_std']),
        'r2_test': float(mejor_reg_r2_test),
        'rmse_test': float(mejor_reg_rmse_test),
        'mae_test': float(fila_mejor_reg['MAE_Test']),
    },
    'target': 'v_out_norm (velocidad normal de salida, m/s)',
    'grupos': 'racket_id',
    'baseline': {
        'r2_test': float(baseline_reg_r2),
        'rmse_test': float(baseline_reg_rmse),
    },
}
ruta_mejor_reg = RUTA_MODELOS / 'mejor_regresor.joblib'
joblib.dump(paquete_reg, ruta_mejor_reg)
print(f'   Mejor regresor guardado: {ruta_mejor_reg}')

# --- ONNX con pipeline completo ---
validacion_onnx = {}
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import DoubleTensorType
    import onnxruntime as rt

    def exportar_pipeline_onnx(pipeline, features, ruta, es_clasificador):
        initial_type = [('float_input', DoubleTensorType([None, len(features)]))]
        options = {id(pipeline): {'zipmap': False}} if es_clasificador else None
        onx = convert_sklearn(pipeline, initial_types=initial_type, options=options)
        with open(ruta, 'wb') as f:
            f.write(onx.SerializeToString())
        return ruta

    ruta_onnx_clf = exportar_pipeline_onnx(
        mejor_clf_pipeline, features_clf, RUTA_MODELOS / 'mejor_clasificador.onnx', True
    )
    ruta_onnx_reg = exportar_pipeline_onnx(
        mejor_reg_pipeline, features_reg, RUTA_MODELOS / 'mejor_regresor.onnx', False
    )
    print(f'   ONNX clasificador: {ruta_onnx_clf}')
    print(f'   ONNX regresor:     {ruta_onnx_reg}')

    sess_clf = rt.InferenceSession(str(ruta_onnx_clf))
    input_clf = sess_clf.get_inputs()[0].name
    salida_clf = sess_clf.run(None, {input_clf: X_test_clf.values.astype(np.float64)})
    pred_onnx_clf = salida_clf[0]
    pred_joblib_clf = mejor_clf_pipeline.predict(X_test_clf)
    acc_onnx_clf = accuracy_score(pred_joblib_clf, pred_onnx_clf)

    sess_reg = rt.InferenceSession(str(ruta_onnx_reg))
    input_reg = sess_reg.get_inputs()[0].name
    pred_onnx_reg = sess_reg.run(None, {input_reg: X_test_reg.values.astype(np.float64)})[0].ravel()
    pred_joblib_reg = mejor_reg_pipeline.predict(X_test_reg)
    max_diff_reg = float(np.max(np.abs(pred_onnx_reg - pred_joblib_reg)))

    validacion_onnx = {
        'clasificador_coincidencia': float(acc_onnx_clf),
        'regresor_max_diferencia': max_diff_reg,
        'nota': 'Predicciones ONNX sobre features RAW con entrada float64; el pipeline incluye StandardScaler.',
    }
    print(f'   Validación ONNX clasificador (coincidencia): {acc_onnx_clf:.4f}')
    print(f'   Validación ONNX regresor (max |dif|):       {max_diff_reg:.6f}')
except ImportError:
    print('\n   skl2onnx/onnxruntime no disponibles. Se omite la exportación ONNX.')
    validacion_onnx = {'error': 'skl2onnx/onnxruntime no disponibles'}
except Exception as exc:
    print(f'\n   No se pudo exportar/validar ONNX: {exc}')
    validacion_onnx = {'error': str(exc)}

# --- Inferencia con modelos guardados sobre muestras reales de test ---
print('\n--- Prueba de inferencia con modelos cargados (5 rebotes reales de test) ---')
clf_cargado = joblib.load(ruta_mejor_clf)
reg_cargado = joblib.load(ruta_mejor_reg)
print(f'   Clasificador cargado: {clf_cargado["nombre"]}')
print(f'   Regresor cargado: {reg_cargado["nombre"]}')

muestras = df_test.sample(n=5, random_state=RANDOM_STATE).copy()
X_nuevas_clf = muestras[clf_cargado['features']]
X_nuevas_reg = muestras[reg_cargado['features']]

pred_clf = clf_cargado['pipeline'].predict(X_nuevas_clf)
pred_reg = reg_cargado['pipeline'].predict(X_nuevas_reg)
if hasattr(clf_cargado['pipeline'], 'predict_proba'):
    proba_clf = clf_cargado['pipeline'].predict_proba(X_nuevas_clf).max(axis=1)
else:
    proba_clf = np.full(len(pred_clf), np.nan)

resultados_inferencia = pd.DataFrame({
    'racket_id': muestras['racket_id'].values,
    'configuracion': muestras['configuracion'].values,
    'familia_real': [etiquetas_clf[v] for v in muestras['familia_bin'].values],
    'familia_predicha': [etiquetas_clf[g] for g in pred_clf],
    'confianza': np.round(proba_clf, 3),
    'v_out_norm_real': np.round(muestras['v_out_norm'].values, 3),
    'v_out_norm_predicha': np.round(pred_reg, 3),
})
resultados_inferencia['error_absoluto'] = np.round(
    np.abs(resultados_inferencia['v_out_norm_real'] - resultados_inferencia['v_out_norm_predicha']), 3
)
resultados_inferencia.to_csv(RUTA_MODELOS / 'resultados_inferencia.csv', index=False)

print('\n' + '=' * 80)
print('  RESULTADOS DE INFERENCIA (5 rebotes reales de test)')
print('=' * 80)
print(resultados_inferencia.to_string(index=False))

# %% [markdown]
# ## 6. Resumen y metadatos

# %%
import sklearn

metadata = {
    'fecha': datetime.now().isoformat(timespec='seconds'),
    'version_proyecto': 'corregida-2.0-real',
    'python': sys.version,
    'librerias': {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__,
        'seaborn': sns.__version__,
        'scikit-learn': sklearn.__version__,
        'joblib': joblib.__version__,
    },
    'dataset': {
        'archivo': RUTA_DATASET.name,
        'ruta': str(RUTA_DATASET),
        'fuente': 'https://github.com/gossardt/tt_rackets',
        'articulo': 'https://arxiv.org/abs/2604.11349',
        'licencia': 'MIT',
        'filas': int(len(df)),
        'columnas': list(df.columns),
        'configuraciones': int(df['racket_id'].nunique()),
        'familias': sorted(df['rubber_family'].unique().tolist()),
        'rebotes_por_configuracion': {
            str(k): int(v) for k, v in df['racket_id'].value_counts().sort_index().items()
        },
    },
    'split': {
        'metodo': 'Holdout por configuración de raqueta (grupos no vistos)',
        'test_groups': TEST_GROUPS,
        'train_groups': sorted(df_train['racket_id'].unique().tolist()),
        'test_groups_presentes': sorted(df_test['racket_id'].unique().tolist()),
        'train': int(len(df_train)),
        'test': int(len(df_test)),
        'grupos_train': int(groups_train.nunique()),
        'grupos_test': int(groups_test.nunique()),
        'cv_folds': CV_FOLDS,
    },
    'diagnostico_fuga': {
        'clasificacion_accuracy_con_metadata_del_target': float(acc_leak),
        'clasificacion_accuracy_sin_metadata_del_target': float(acc_no),
        'regresion_r2_con_cor': float(r2_leak),
        'regresion_r2_sin_variables_de_salida': float(r2_no),
        'nota': (
            'La clasificación excluye familia_goma, rubber_type y racket_id. '
            'La regresión excluye v_out_norm, cor, spin_out y demás variables posteriores al rebote.'
        ),
    },
    'clasificacion': {
        'target': 'familia_bin (Invertida vs No invertida)',
        'etiquetas': etiquetas_clf,
        'features': features_clf,
        'baseline': {
            'f1_cv': float(baseline_clf_f1_cv),
            'f1_test': float(baseline_clf_f1_test),
        },
        'mejor': {
            'modelo': mejor_clf_nombre,
            'f1_cv': float(mejor_clf_f1_cv),
            'f1_cv_std': float(fila_mejor_clf['F1_CV_std']),
            'f1_test': float(mejor_clf_f1_test),
            'accuracy_test': float(fila_mejor_clf['Accuracy_Test']),
            'parametros': fila_mejor_clf['Mejores_Parametros'],
        },
    },
    'regresion': {
        'target': 'v_out_norm (velocidad normal de salida, m/s)',
        'features': features_reg,
        'baseline': {
            'r2_test': float(baseline_reg_r2),
            'rmse_test': float(baseline_reg_rmse),
        },
        'mejor': {
            'modelo': mejor_reg_nombre,
            'r2_cv': float(mejor_reg_r2_cv),
            'r2_cv_std': float(fila_mejor_reg['R2_CV_std']),
            'r2_test': float(mejor_reg_r2_test),
            'rmse_test': float(mejor_reg_rmse_test),
            'parametros': fila_mejor_reg['Mejores_Parametros'],
        },
    },
    'validacion_onnx': validacion_onnx,
    'limitaciones': [
        'El dataset es real y fue medido experimentalmente, pero cubre solo 10 configuraciones de raqueta.',
        'El movimiento se restringe a un plano 2D; no se modela la dinámica tridimensional completa.',
        'El rango de velocidades y spin está limitado por la máquina lanzapelotas.',
        'El test contiene solo dos configuraciones (R02 y R05); los resultados no se extrapolan a todas las raquetas.',
        'La clasificación usa la respuesta dinámica posterior al rebote; no es un predictor previo al golpe.',
        'La regresión usa metadata ordinal de la raqueta (rigidez y familia de goma), no mediciones directas de dureza o adherencia.',
    ],
}
ruta_metadata = RUTA_MODELOS / 'metadata.json'
with open(ruta_metadata, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f'\n   Metadatos exportados: {ruta_metadata}')

print('\n' + '=' * 80)
print('  RESUMEN FINAL DEL PROYECTO CORREGIDO CON DATOS REALES')
print('=' * 80)
print(f"""
  Dataset: {len(df)} rebotes | {df['racket_id'].nunique()} configuraciones de raqueta
  Train: {len(df_train)} rebotes ({groups_train.nunique()} configuraciones)
  Test:  {len(df_test)} rebotes ({groups_test.nunique()} configuraciones: {TEST_GROUPS})
  Clasificación: 11 modelos | target: familia_bin
  Mejor: {mejor_clf_nombre}
     F1 CV:   {mejor_clf_f1_cv:.4f} +/- {fila_mejor_clf['F1_CV_std']:.4f}
     F1 Test: {mejor_clf_f1_test:.4f}
     Accuracy Test: {fila_mejor_clf['Accuracy_Test']:.4f}
  Regresión: 11 modelos | target: v_out_norm
  Mejor: {mejor_reg_nombre}
     R2 CV:   {mejor_reg_r2_cv:.4f} +/- {fila_mejor_reg['R2_CV_std']:.4f}
     R2 Test: {mejor_reg_r2_test:.4f}
     RMSE Test: {mejor_reg_rmse_test:.4f}
""")
print('=' * 80)
print('  Proceso completado. Revisa graficos/ y modelos/.')
print('=' * 80)
