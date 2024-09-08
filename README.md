# Optimización Computacional de Vision Transformer (ViT) para el Cribado Eficiente del Glaucoma

Las redes Vision Transformer (ViT) son eficaces en el procesamiento de imágenes, pero su alto costo computacional limita su uso en entornos con recursos limitados. Esta investigación optimiza la arquitectura ViT para el cribado del glaucoma, ajustando parámetros clave y aplicando CLAHE para mejorar las imágenes. Logramos reducir el tiempo de entrenamiento en un 65% y alcanzar un accuracy del 87.11%, demostrando que la arquitectura optimizada es eficiente y viable para aplicaciones clínicas.

Data set utilizado: https://www.kaggle.com/datasets/sabari50312/fundus-pytorch

<h2>Resultados Preliminares:</h2>
Para el desarrollo del nuevo modelo ViT personalizado, se ajustaron los hiperparámetros mencionados anteriormente y se probaron diferentes configuraciones para encontrar un diseño más eficiente en términos de costo computacional y rendimiento. Se utilizaron métricas como accuracy, precision, recall, y F1-score para evaluar los desempeños de los modelos.

Los resultados preliminares identificaron una arquitectura ViT optimizada que reduce significativamente los recursos computacionales necesarios sin sacrificar la eficiencia.

| Experimentos      | Modelo Base ViT B-16 | Modelo Optimizado |
|-------------------|----------------------|-------------------|
| **Patch size**     | 16                   | 16                |
| **Dim. Proy.**     | 768                  | 128               |
| **Head Att.**      | 12                   | 6                 |
| **T. Units**       | 3072, 768            | 256, 128          |
| **T. Layers**      | 12                   | 8                 |
| **MLP head**       | 3072, 768            | 1024, 512         |
| **Hiperparámetros**| 86,014,114           | 30,168,194        |
| **Tiempo/época**   | 3 horas              | 30 min.           |
| **Accuracy**       | 85.82%               | 87.11%            |
| **Precisión**      | 61.21%               | 65.28%            |
| **Recall**         | 97.37%               | 88.71%            |
| **F1-score**       | 75.17%               | 75.21%            |

*Tabla comparativa de Arquitecturas. Dim. Proy.: Dimensión de Proyección, Head Att.: Cabezales de atención, T. Units: Unidades Transformers, T. Layers: Capas Transformer.*
