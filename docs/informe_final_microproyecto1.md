# **Clasificación de MRI Cerebral con Transfer Learning: EfficientNet-B0 y ResNet50** 

_Sergio Pardo, Jefferson Moreno, John Pino_ Maestría en Inteligencia Artificial — Técnicas de Deep Learning — Abril 2026 

## **I. INTRODUCCIÓN** 

La clasificación automática de tumores cerebrales en imágenes de resonancia magnética (MRI) es un problema de clasificación supervisada con alto impacto en el diagnóstico clínico. La revisión manual es costosa, presenta variabilidad entre evaluadores y requiere alta especialización, además de enfrentar desafíos como ruido en las imágenes y variabilidad anatómica entre pacientes. 

El objetivo de este trabajo es comparar el desempeño de modelos basados en _transfer learning_ para clasificar imágenes 2D de MRI en cuatro categorías: glioma, meningioma, pituitary y healthy, evaluando su balance entre exactitud y generalización. El _transfer learning_ permite adaptar representaciones visuales preentrenadas a dominios médicos con datasets moderados. En este contexto, EfficientNet-B0 [1] propone un escalado compuesto que maximiza la precisión con menor costo computacional, mientras ResNet50 [2] utiliza conexiones residuales que facilitan el entrenamiento profundo y han demostrado buen desempeño en tareas de imágenes médicas al permitir especializar capas superiores mediante _fine-tuning_ . En este trabajo se comparan tres configuraciones —EfficientNet-B0, ResNet50 y ResNet50 con _fine-tuning_ — utilizando métricas de evaluación estándar para determinar cuál ofrece el mejor desempeño dentro del problema planteado. 

## **II. METODOLOGÍA** 

## **_A. Enfoque general_** 

Se propone un pipeline de clasificación de imágenes 2D de MRI basado en _transfer learning_ . El uso de cortes bidimensionales reduce la carga computacional frente a enfoques 3D y permite aplicar CNNs preentrenadas en ImageNet. El pipeline incluye: (1) preprocesamiento y partición del dataset, (2) adaptación de arquitecturas a un problema de 4 clases y (3) entrenamiento con evaluación en test independiente. Se comparan EfficientNet-B0 [1] y ResNet50 [2], esta última también con _fine-tuning_ . Ambas se eligieron por ser referencias consolidadas en clasificación médica con balance comprobado entre capacidad representacional y eficiencia computacional 

## **_B. Dataset y preprocesamiento_** 

incluyó: eliminación de duplicados (hash MD5), redimensionamiento a 224×224, conversión a RGB y normalización con estadísticas de ImageNet. La partición fue estratificada (80/10/10, semilla 42). Se aplicó _data augmentation_ solo en entrenamiento (volteo horizontal, rotación ±15°, variación de contraste) para mejorar la generalización. 

## **_C. Arquitectura del modelo_** 

Ambos modelos se inicializaron con pesos de ImageNet y se adaptaron a 4 clases (softmax). 

**EfficientNet-B0:** (5.3M parámetros) extractor congelado y capa densa final (1.280→4), entrenando ~5K parámetros. **ResNet50:** (25M parámetros) en _transfer learning_ se congeló el extractor y se añadió Dropout(0.2) + capa densa. En _fine-tuning_ se descongelaron los bloques _conv5_ , manteniendo BatchNorm congeladas para estabilidad. Este enfoque aprovecha representaciones preentrenadas y reduce la necesidad de grandes datasets. 

## **_D. Entrenamiento y evaluación_** 

Todos los modelos se entrenaron con los mismos hiperparámetros base: optimizador Adam, función de pérdida SparseCategoricalCrossentropy, tamaño de lote de 32 y 10 épocas. La tasa de aprendizaje fue 1×10⁻³ para la fase de transfer learning y 1×10 para el fine-tuning, valor⁻⁵ reducido para ajustar gradualmente las capas superiores sin destruir las representaciones preentrenadas. En cada configuración se guardó el modelo con menor pérdida de validación (val\_loss) —criterio preferido sobre val_accuracy por ser más robusto ante incertidumbre entre clases—, usando un conjunto de validación independiente del conjunto de prueba. La evaluación final se realizó sobre el conjunto de prueba (10%), que no participó en ningún proceso de selección de modelo. Se reportan accuracy y cross-entropy loss como métricas globales, y precisión, recall y F1-score por clase. Este último se justifica porque en clasificación médica el accuracy global puede ocultar fallos críticos en clases específicas: un modelo con 90% de accuracy pero bajo recall en meningioma implica falsos negativos clínicamente inaceptables. 

## **III. RESULTADOS** 

## **_A. Resultados cuantitativos_** 

La Tabla I compara los tres modelos en test. ResNet50+FT obtuvo la mejor exactitud (96.67%) y menor loss (0.1172), superando en 7.43 pp a EfficientNet-B0 y 3.94 pp a ResNet50. La Tabla II muestra el F1 por clase: meningioma fue la más difícil en los tres enfoques, con mejora relevante tras el fine-tuning (0.7889 → 0.9333). Healthy y pituitary alcanzaron los mejores resultados consistentemente. 

Se utilizó el dataset Brain Tumor MRI Scans [3] (7.023 imágenes, 4 clases balanceadas). El preprocesamiento 

1 

**TABLA I. Comparación de resultados en el conjunto de** **<u>prueba.</u>** 

|**Modelo**|**Loss**|**Acc.**|**F1 mac.**|
|---|---|---|---|
|EfficientNet-B0|0.2885|89.24%|0.8878|
|ResNet50|0.2049|92.73%|0.9253|
|ResNet50+FT|0.1172|96.67%|0.9657|



**TABLA II. F1-score por clase en el conjunto de prueba.** 

|**Clase**|**EffB0**|**RN50**|**RN50+FT**|**Mejor**|
|---|---|---|---|---|
|glioma|0.8393|0.9185|0.9657|RN50+FT|
|healthy|0.9647|0.9600|0.9730|RN50+FT|
|meningioma|0.7889|0.8508|0.9333|RN50+FT|
|pituitary|0.9189|0.9720|0.9908|RN50+FT|



## **_B. Resultados cualitativos_** 

La Fig. 1 presenta las curvas de aprendizaje de EfficientNet-B0. La Fig. 2 muestra las curvas de aprendizaje de ResNet50 (transfer learning + fine-tuning). La Fig. 3 compara predicciones correctas e incorrectas entre los tres modelos; los fallos disminuyen progresivamente de EfficientNet-B0 a ResNet50+FT, concentrándose en imágenes con características visuales ambiguas entre meningioma y glioma. La Fig. 4 compara las matrices de confusión de los tres modelos: los errores se concentran en meningioma, confundida principalmente con glioma. 



_Fig. 1. Curvas de aprendizaje de EfficientNet-B0 con 10 épocas_ 



_Fig. 2. Curvas de aprendizaje de ResNet50: transfer learning (1-10) y fine-tuning (11-20). La línea roja marca el inicio del fine-tuning_ 





_Fig. 4. Matrices de confusión en test: EfficientNet-B0 (izq.), ResNet50 (centro) y ResNet50+FT (der.)._ 

**TABLA III.** Desglose de Verdaderos/Falsos Positivos y Negativos por arquitectura y tipo de lesión 

|**Modelo**|**Clase**|**VP**|**VN**|**FN**|**FP**|
|---|---|---|---|---|---|
||Glioma|141|480|21|18|
|**EfficientNet-B0**|Healthy|164|484|7|5|
||Meningioma|114|485|39|22|
||Pituitary|170|460|4|26|
||Glioma|143|490|19|8|
|**ResNet50**|Healthy|165|487|6|2|
||Meningioma|134|479|19|28|
||Pituitary|170|476|4|10|
||Glioma|151|495|11|3|
|**ResNet50 + FT**|Healthy|170|487|1|2|
||Meningioma|147|492|6|15|
||Pituitary|170|484|4|2|



## **IV. DISCUSIÓN** 

La mejora progresiva entre enfoques responde a causas teóricas concretas. EfficientNet-B0 **[1]** presentó mayor sesgo relativo (underfitting, val_acc ≈0.89) dado su menor capacidad (5.3M parámetros). ResNet50 **[2]** , con 25M parámetros y conexiones residuales, extrajo representaciones más discriminativas (+3.49 pp). El finetuning de conv5 permitió especializar características de alto nivel al dominio MRI, explicando el salto de +3.94 pp adicionales. Al final del entrenamiento, ResNet50+FT mostró brecha train/val (99.17% vs 97.27%), indicando ajuste fuerte, pero sin deterioro grave —la selección por val_loss en época 7 contuvo el sobreajuste. La clase meningioma fue sistemáticamente la más difícil por su variabilidad morfológica en MRI; el fine-tuning elevó su recall de 0.7451 a 0.9608, crítico para reducir falsos negativos clínicos. Limitación principal: la partición a nivel de imagen (no de paciente) puede sobreestimar la generalización. 

Como trabajo futuro, se propone realizar partición por paciente, early stopping dinámico y validación clínica vía Grad-CAM. 

_Fig. 3. Comparación de predicciones correctas (fila sup.) e incorrectas (fila inf.) entre EfficientNet-B0, ResNet50 y ResNet50+FT (izq. a der.)._ 

2 

# **REFERENCIAS** 

[1] M. Tan y Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” _Proc. ICML_ , 2019. Disponible: https://research.google/pubs/efficientnetrethinking-model-scaling-for-convolutionalneural-networks/ 

[2] K. He et al., “Deep Residual Learning for Image Recognition,” _Proc. CVPR_ , 2016. Disponible: https://arxiv.org/abs/1512.03385 

[3] R. Mandal, “Brain Tumor (MRI Scans),” _Kaggle_ , 2023. Disponible: https://www.kaggle.com/datasets/rm1000/braintumor-mri-scans 

3 

