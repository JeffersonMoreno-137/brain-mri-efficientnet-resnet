# Clasificación de MRI Cerebral con Transfer Learning: EfficientNet-B0 y ResNet50

Este repositorio contiene la implementación, experimentación y análisis cuantitativo y cualitativo de modelos de Deep Learning basados en Transfer Learning y Fine-Tuning para la clasificación multiclase de resonancias magnéticas cerebrales (MRI). El trabajo evalúa y compara el desempeño de EfficientNet-B0 y ResNet50 sobre el dataset Brain Tumor MRI Scans, adaptando representaciones preentrenadas en ImageNet al dominio clínico.

Autores: Sergio Pardo, Jefferson Moreno, John Pino  
Maestría en Inteligencia Artificial — Técnicas de Deep Learning

---

## Contenido

1. [Introducción y Contexto Clínico](#introducción-y-contexto-clínico)
2. [Dataset y Pipeline de Datos](#dataset-y-pipeline-de-datos)
3. [Arquitecturas y Estrategia de Transfer Learning](#arquitecturas-y-estrategia-de-transfer-learning)
4. [Configuración de Entrenamiento](#configuración-de-entrenamiento)
5. [Resultados Cuantitativos](#resultados-cuantitativos)
6. [Resultados Cualitativos y Curvas de Aprendizaje](#resultados-cualitativos-y-curvas-de-aprendizaje)
7. [Matrices de Confusión y Diagnóstico Clínico](#matrices-de-confusión-y-diagnóstico-clínico)
8. [Discusión y Limitaciones](#discusión-y-limitaciones)
9. [Estructura del Proyecto](#estructura-del-proyecto)
10. [Instrucciones de Instalación y Uso](#instrucciones-de-instalación-y-uso)
11. [Referencias](#referencias)

---

## Introducción y Contexto Clínico

La clasificación automatizada de neoplasias intracraneales mediante resonancia magnética (MRI) constituye una herramienta clave de apoyo al diagnóstico médico. La evaluación visual por parte de especialistas es costosa, consume tiempo significativo y presenta variabilidad inter-observador, además de lidiar con factores como artefactos técnicos, ruido de adquisición y heterogeneidad anatómica entre pacientes.

El propósito de esta investigación es determinar el balance óptimo entre costo computacional, capacidad de discriminación morfológica y capacidad de generalización en cortes bidimensionales de MRI. Para ello, se evalúan tres aproximaciones:

1. **EfficientNet-B0:** Modelo optimizado mediante escalado compuesto con ~5.3 millones de parámetros, enfocado en eficiencia de recursos.
2. **ResNet50 (Transfer Learning):** Arquitectura residual de 25 millones de parámetros con extractor congelado, que aprovecha representaciones genéricas de bajo y medio nivel.
3. **ResNet50 + Fine-Tuning:** Descongelamiento de las capas convolucionales superiores (bloque `conv5_`) para especializar los filtros abstractos a las texturas y bordes particulares de las lesiones cerebrales.

---

## Dataset y Pipeline de Datos

Se utilizó el dataset público **Brain Tumor (MRI Scans)**, compuesto por 7,023 imágenes axiales distribuidas en cuatro clases diagnósticas:

- `glioma`: Tumores gliales con patrones infiltrativos y márgenes difusos.
- `healthy`: Controles sanos sin hallazgos patológicos tumorales.
- `meningioma`: Tumores de origen meníngeo, habitualmente extraaxiales con realce homogéneo.
- `pituitary`: Adenomas hipofisarios ubicados en la fosa selar.

![Muestras del Dataset](figures/02_mri_samples.png)

### Preprocesamiento y Limpieza

- **Filtrado de archivos no válidos:** Eliminación de archivos ocultos del sistema y verificación de integridad de extensiones (`.jpg`, `.jpeg`, `.png`).
- **Eliminación de duplicados exactos:** Identificación y descarte de copias idénticas a través del cálculo de hash MD5 sobre el flujo binario de cada archivo.
- **Normalización y escalado espacial:** Redimensionamiento bilineal uniforme a 224x224 píxeles con 3 canales (RGB) y normalización adaptada a las estadísticas del modelo base (ImageNet).
- **Partición estratificada:** División del conjunto total en proporciones 80% entrenamiento, 10% validación y 10% prueba independiente (semilla 42), conservando la prevalencia de cada patología en todas las particiones.
- **Data Augmentation:** Aplicado exclusivamente en la etapa de entrenamiento mediante transformaciones geométricas y fotométricas moderadas (volteo horizontal aleatorio, rotación de hasta +/- 15 grados y variación controlada de contraste), preservando la coherencia anatómica sin generar artificios inverosímiles.

![Distribución de Clases](figures/01_dataset_distribution.png)

---

## Arquitecturas y Estrategia de Transfer Learning

Ambas familias se adaptaron para clasificar las 4 categorías patológicas mediante capas densas finales con función de activación Softmax.

```
Pipeline de Clasificación:
Entrada (224x224x3) -> Augmentation (Train) -> Preprocesamiento ImageNet -> Backbone Preentrenado -> GlobalAveragePooling2D -> Dropout(0.2) -> Dense(4, Softmax)
```

### 1. EfficientNet-B0
- **Total de parámetros:** ~5.3M.
- **Estrategia:** Extractor convolucional congelado; únicamente se entrena la cabeza de clasificación (1,280 -> 4), con ~5,124 parámetros entrenables.

### 2. ResNet50 (Transfer Learning)
- **Total de parámetros:** ~25M.
- **Estrategia:** Backbone congelado; capa `GlobalAveragePooling2D` conectada a `Dropout(0.2)` y `Dense(4, Softmax)`.

### 3. ResNet50 con Fine-Tuning
- **Estrategia:** Tomando como punto de partida los pesos del mejor checkpoint de la etapa de Transfer Learning, se descongelan todas las capas correspondientes al bloque `conv5_x`.
- **Estabilidad de capas Batch Normalization:** Las capas `BatchNormalization` se mantienen expresamente congeladas (`trainable = False`) para evitar distorsiones en las estadísticas de media y varianza móvil calculadas sobre ImageNet.

---

## Configuración de Entrenamiento

Todos los esquemas fueron entrenados bajo condiciones homogéneas para garantizar una comparación equitativa:

- **Optimizador:** Adam.
- **Función de pérdida:** Sparse Categorical Cross-Entropy.
- **Tamaño de lote (batch size):** 32.
- **Tasa de aprendizaje (Learning Rate):**
  - Fase de Transfer Learning: 1e-3.
  - Fase de Fine-Tuning: 1e-5 (reducción de dos órdenes de magnitud para impedir el olvido catastrófico de características preentrenadas).
- **Épocas:** 10 épocas en cada fase.
- **Criterio de parada y selección de checkpoint:** Monitoreo estricto de la menor pérdida de validación (`val_loss`), seleccionado sobre `val_accuracy` por penalizar la sobreconfianza en predicciones erróneas.
- **Hardware de ejecución:** Aceleración por GPU/MPS con asignación dinámica de memoria y prefetch paralelo con `tf.data.AUTOTUNE`.

---

## Resultados Cuantitativos

La evaluación definitiva se llevó a cabo sobre el conjunto de prueba independiente (10% del dataset original, 703 imágenes), el cual permaneció completamente aislado de las fases de entrenamiento y ajuste de hiperparámetros.

### Tabla I. Comparación de Rendimiento Global en Conjunto de Prueba

| Modelo | Test Loss | Test Accuracy | Macro F1-Score | Parámetros Entrenables |
|:---|:---:|:---:|:---:|:---:|
| EfficientNet-B0 | 0.2885 | 89.24% | 0.8878 | ~5 K |
| ResNet50 (TL) | 0.2049 | 92.73% | 0.9253 | ~8 K |
| **ResNet50 + Fine-Tuning** | **0.1172** | **96.67%** | **0.9657** | **~15 M** |

ResNet50 con Fine-Tuning superó en **7.43 puntos porcentuales** a EfficientNet-B0 y en **3.94 puntos porcentuales** a ResNet50 con Transfer Learning base, alcanzando una pérdida de prueba de 0.1172.

![Comparación Global de Accuracy](figures/16_models_accuracy_comparison.png)

### Tabla II. Rendimiento Detallado por Clase (F1-Score)

| Clase Patológica | EfficientNet-B0 | ResNet50 (TL) | ResNet50 + Fine-Tuning | Mejor Configuración |
|:---|:---:|:---:|:---:|:---:|
| `glioma` | 0.8393 | 0.9185 | **0.9657** | ResNet50 + FT |
| `healthy` | 0.9647 | 0.9600 | **0.9730** | ResNet50 + FT |
| `meningioma` | 0.7889 | 0.8508 | **0.9333** | ResNet50 + FT |
| `pituitary` | 0.9189 | 0.9720 | **0.9908** | ResNet50 + FT |

---

## Resultados Cualitativos y Curvas de Aprendizaje

### Curvas de Aprendizaje: EfficientNet-B0

EfficientNet-B0 exhibe convergencia regular pero limitada por la baja dimensionalidad del espacio de adaptación; el modelo converge tempranamente hacia una exactitud de validación cercana al 89%, reflejando un ligero subajuste (underfitting) relativo al dominio de resonancias magnéticas.

![Curvas EfficientNet-B0](figures/03_efficientnet_learning_curves.png)

### Curvas de Aprendizaje: ResNet50 (Transfer Learning y Fine-Tuning)

La evolución completa de ResNet50 demuestra la efectividad de la transición en dos fases. Durante las primeras 10 épocas (red base congelada), la pérdida desciende sostenidamente. Tras habilitar el fine-tuning en la época 10 con una tasa de aprendizaje de 1e-5, el modelo experimenta una reducción adicional pronunciada en la función de pérdida sin presentar divergencia o inestabilidad numérica.

![Curvas Combinadas ResNet50](figures/06_resnet50_vs_ft_combined.png)

---

## Matrices de Confusión y Diagnóstico Clínico

El desglose de clasificaciones correctas e incorrectas en el conjunto de prueba evidencia la concentración de incertidumbre entre pares patológicos con similitud de textura tisular.

![Matrices de Confusión Comparativas](figures/composite_confusion_matrices.png)

*De izquierda a derecha: Matrices de confusión en prueba para EfficientNet-B0, ResNet50 y ResNet50 + Fine-Tuning.*

### Tabla III. Desglose de Matriz Diagnóstica por Arquitectura y Tipo de Lesión

| Modelo | Patología | Verdaderos Positivos (VP) | Verdaderos Negativos (VN) | Falsos Negativos (FN) | Falsos Positivos (FP) |
|:---|:---|:---:|:---:|:---:|:---:|
| **EfficientNet-B0** | Glioma | 141 | 480 | 21 | 18 |
| | Healthy | 164 | 484 | 7 | 5 |
| | Meningioma | 114 | 485 | 39 | 22 |
| | Pituitary | 170 | 460 | 4 | 26 |
| **ResNet50 (TL)** | Glioma | 143 | 490 | 19 | 8 |
| | Healthy | 165 | 487 | 6 | 2 |
| | Meningioma | 134 | 479 | 19 | 28 |
| | Pituitary | 170 | 476 | 4 | 10 |
| **ResNet50 + FT** | **Glioma** | **151** | **495** | **11** | **3** |
| | **Healthy** | **170** | **487** | **1** | **2** |
| | **Meningioma** | **147** | **492** | **6** | **15** |
| | **Pituitary** | **170** | **484** | **4** | **2** |

### Inspección Visual de Aciertos y Fallos

El análisis de casos individuales revela que los falsos negativos en meningioma corresponden preponderantemente a imágenes donde el tumor se proyecta cerca del parénquima cerebral sin hiperintensidad nítida, lo que confunde a los modelos con gliomas de bajo grado. Con el fine-tuning, el modelo adquiere filtros capaces de resolver la interfaz aracnoidea y la base dural, reduciendo los falsos negativos de 39 a solo 6.

![Predicciones Cualitativas ResNet50 FT](figures/15_resnet50_ft_qualitative.png)

---

## Discusión y Limitaciones

1. **Capacidad Representacional y Especialización:** ResNet50 ofrece una base de extracción con mayor expresividad respecto a EfficientNet-B0 para imágenes médicas con cortes axiales complejos. El ajuste fino del bloque `conv5_` resulta indispensable para adaptar los núcleos convolucionales de alto nivel a las morfologías oncológicas.
2. **Impacto en Falsos Negativos Clínicos:** En la práctica médica, un falso negativo en patología tumoral conlleva implicaciones severas. ResNet50 + FT elevó el recall en meningioma de 0.7451 a 0.9608, consolidando una frontera de decisión sustancialmente más confiable.
3. **Control del Sobreajuste:** Aunque al cierre del entrenamiento la brecha train/val fue de 99.17% contra 97.27%, la preservación del checkpoint con mínima pérdida de validación en la época 7 contuvo la degradación del poder de generalización.
4. **Limitación Metodológica:** La partición del dataset se estructuró a nivel de corte individual en lugar de a nivel de paciente. Esta condición, habitual en benchmarks públicos sin metadatos clínicos adicionales, puede inducir una ligera sobreestimación de la generalización en entornos de distribución libre.

---

## Estructura del Proyecto

```
brain-mri-efficientnet-resnet/
├── README.md                          # Documentación técnica completa del proyecto
├── LICENSE                            # Licencia MIT
├── requirements.txt                   # Dependencias reproducibles de Python
├── main.py                            # Punto de entrada por línea de comandos
├── src/                               # Código modular de la solución
│   ├── __init__.py                    # Módulo principal
│   ├── config.py                      # Hiperparámetros, rutas y semillas aleatorias
│   ├── data.py                        # Carga, verificación MD5, partición y tf.data pipeline
│   ├── models.py                      # Definición de EfficientNet-B0, ResNet50 y fine-tuning
│   ├── train.py                       # Bucles de entrenamiento y callbacks de guardado
│   └── evaluate.py                    # Cálculo de métricas, reportes y matrices de confusión
├── notebooks/                         # Cuadernos interactivos
│   └── brain_mri_classification.ipynb # Cuaderno reproducible paso a paso
├── figures/                           # Gráficos, curvas y matrices de resultados
│   ├── 01_dataset_distribution.png
│   ├── 02_mri_samples.png
│   ├── 03_efficientnet_learning_curves.png
│   ├── 04_resnet50_learning_curves.png
│   ├── 05_resnet50_ft_learning_curves.png
│   ├── 06_resnet50_vs_ft_combined.png
│   ├── 07_efficientnet_metrics_bar.png
│   ├── 08_efficientnet_confusion_matrix.png
│   ├── 09_efficientnet_qualitative.png
│   ├── 10_resnet50_metrics_bar.png
│   ├── 11_resnet50_confusion_matrix.png
│   ├── 12_resnet50_qualitative.png
│   ├── 13_resnet50_ft_metrics_bar.png
│   ├── 14_resnet50_ft_confusion_matrix.png
│   ├── 15_resnet50_ft_qualitative.png
│   ├── 16_models_accuracy_comparison.png
│   └── composite_confusion_matrices.png
└── docs/                              # Artículos, notas y referencias de respaldo
    ├── informe_final_microproyecto1.md
    └── cnn_mri_pytorch_reference.md
```

---

## Instrucciones de Instalación y Uso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/JeffersonMoreno-137/brain-mri-efficientnet-resnet.git
cd brain-mri-efficientnet-resnet
```

### 2. Crear y Activar un Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Estructura del Dataset

Descargue el dataset de Kaggle (Rajarshi Mandal: Brain Tumor MRI Scans) y ubíquelo en la raíz bajo la carpeta `dataset/`:

```
dataset/
├── glioma/
├── healthy/
├── meningioma/
└── pituitary/
```

### 5. Ejecutar Entrenamiento y Evaluación

Entrenar y evaluar todos los modelos:

```bash
python main.py --model all
```

Ejecutar únicamente ResNet50 con Fine-Tuning:

```bash
python main.py --model resnet50 --fine-tune
```

Ejecutar mediante Jupyter Notebook:

```bash
jupyter notebook notebooks/brain_mri_classification.ipynb
```

---

## Referencias

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML). https://arxiv.org/abs/1905.11946
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/1512.03385
3. Mandal, R. (2023). *Brain Tumor (MRI Scans) Dataset*. Kaggle. https://www.kaggle.com/datasets/rm1000/braintumor-mri-scans
