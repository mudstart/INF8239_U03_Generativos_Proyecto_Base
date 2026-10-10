# Model Card · Autoencoder y VAE

## Modelos y versiones
- **Autoencoder (AE)**, guardado en `models/autoencoder.keras`, y **decoder del VAE**, en `models/decoder.keras`. El encoder del VAE no se guarda en disco; el script lo usa durante la misma ejecución para evaluar sobre el conjunto de prueba.
- Proyecto `inf8239-u03-generative` v0.1.0. Python 3.12, TensorFlow 2.21.0, Keras 3.15.1 y Streamlit 1.64.0.
- Entrenados con `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` y semilla 42. Las ejecuciones repetidas (terminal y `notebooks/ae_vae_fashion_mnist.ipynb`) reprodujeron exactamente las mismas pérdidas.

## Uso previsto y usuarios
- Demostración académica del curso INF-8239 (Unidad 03): comparar reconstrucción y generación, y explorar el espacio latente 2D del VAE con `app/streamlit_app.py`.
- Usuarios: estudiantes y docentes del curso.

## Usos fuera de alcance
- Generar imágenes realistas o de calidad para producción.
- Aumentar datos para entrenar otros modelos.
- Sustituir datos reales por motivos de privacidad: una muestra sintética no es privada ni original por el hecho de ser sintética.
- Usarlo con imágenes que no sean de Fashion-MNIST (otros tamaños, en color o de otros dominios).
- Generar con el AE: su espacio latente no está regularizado.

## Dataset, licencia y particiones
- **Fashion-MNIST** (Zalando), licencia MIT. Imágenes de 28×28 en escala de grises, 10 clases.
- **Preprocesamiento:** normalización a [0,1] con un canal añadido. `validate_images` comprueba la forma y el rango.
- **Particiones:**

| Partición | AE | VAE |
|---|---|---|
| Entrenamiento | 54,000 | 54,000 |
| Validación | 6,000 (`validation_split=0.1`, early stopping) | 6,000 (las mismas, early stopping) |
| Prueba | 10,000 reservadas; 16 usadas en el panel (semilla 42) | igual |

- **Evaluación final:** las métricas de fidelidad se calculan sobre las 10,000 imágenes de prueba en el script y se presentan en el notebook ejecutado.

## Arquitectura y espacio latente
| Modelo | Arquitectura | Latente | Parámetros |
|---|---|---|---|
| AE | 784 → Dense(16, ReLU) → Dense(784, sigmoid) | 16D determinista | 25,888 |
| Encoder del VAE | 784 → Dense(128, ReLU) → `z_mean`, `z_log_var` | 2D probabilístico | 100,996 |
| Decoder del VAE | 2 → Dense(128, ReLU) → Dense(784, sigmoid) | — | 101,520 |

- El VAE usa el truco de reparametrización `z = μ + exp(0.5·logσ²)·ε`, con una pérdida BCE sumada más la KL frente a N(0, I).
- El latente 2D separa el calzado (botín, sandalia, zapatilla) de la ropa; pantalón y vestido forman regiones propias, pero camiseta, camisa, pulóver y abrigo se solapan.
- Los ejes del latente **no tienen un significado garantizado**.
- El AE y el VAE **no son directamente comparables**: los latentes y las arquitecturas son distintos.

## Métricas de reconstrucción y generación
| Métrica | AE | VAE |
|---|---|---|
| **BCE/píxel · prueba (10,000)** | **0.3068** | **0.3417** |
| **MSE/píxel · prueba (10,000)** | **0.0203** | **0.0333** |
| **SSIM · prueba (10,000; 1 = idéntica)** | **0.5363** | **0.4206** |
| BCE de entrenamiento (última época) | 0.3060 por píxel | 268.10 sumada (≈0.342 por píxel) |
| Pérdida de validación | **0.3071** BCE/píxel (mejor punto, época 6) | **276.55** BCE sumada + KL (mejor punto, época 8) |
| KL (última época, entrenamiento / validación) | — | 6.22 / 6.12 (estable en torno a 6.2) |
| Sobreajuste | No: las curvas convergen y la BCE de prueba ≈ val_loss | No: la validación desciende en las 8 épocas junto al entrenamiento |

- **Reconstrucción:** se conservan las siluetas y se pierden las texturas. Fallan las sandalias (pasan a botín o zapatilla) y los bolsos (pasan a rectángulos difusos o a algo parecido a un pulóver). Por clase en prueba, el bolso es el mayor error del VAE (MSE ≈0.062, el doble que el resto) y la sandalia el del AE.
- **Generación:** de 16 muestras de N(0, I), la mayoría son prendas reconocibles pero borrosas. Dos son híbridos sin forma clara.
- **Interpolación:** continua (botín → camisa → pantalón), con fotogramas intermedios poco plausibles.

## Diversidad y vecinos cercanos
- **Diversidad baja:** unas 4 formas en total, y 5 de las 16 muestras son prácticamente la misma zapatilla. No aparecen bolsos, sandalias ni vestidos.
- **Diversidad cuantificada:** la distancia media entre pares de 500 muestras generadas es el **53 %** de la que hay entre 500 imágenes reales de prueba.
- **Sin colapso del latente:** la KL se mantiene en torno a 6.2, lejos de 0, y en prueba se reparte entre las dos dimensiones (≈3.0 y ≈3.1).
- **Vecinos más cercanos** (distancia euclidiana por píxeles sobre las 60,000 de entrenamiento): distancias de 1.55 a 3.69 (mediana 2.23), con solo 12 vecinos distintos para 16 muestras.
- **Memoria:** la mediana de las muestras (2.23) es menor que la de imágenes reales de prueba al entrenamiento (3.72). No se interpreta como copia: las muestras son borrosas y de bajo contraste, el tipo de imagen que la distancia por píxeles favorece. Ninguna muestra está más cerca que la imagen real más cercana (1.55 frente a 1.39) y no hay duplicados visibles. No hay evidencia de memorización, pero tampoco garantía de privacidad.
- **Limitación de la métrica:** favorece las imágenes oscuras. El mismo vecino (índice 11040) se repite 4 veces, y algunas clases no coinciden con la forma generada.

## Hardware, tiempo y tamaño
- **Hardware:** CPU AMD Ryzen 5 5600X (6 núcleos / 12 hilos), 32 GB de RAM, Windows 11 Pro nativo, sin GPU (TensorFlow ≥ 2.11 no admite GPU en Windows nativo).
- **Tiempo de entrenamiento:** AE ≈4.5–6.2 s y VAE ≈7.7–12.3 s, según la ejecución (la ejecución desde el notebook es la más lenta; sus tiempos quedan en `training_metrics.json`).
- **Tamaño en disco:** `autoencoder.keras` ≈339 KB y `decoder.keras` ≈426 KB.
- **Inferencia:** inmediata en CPU. La app solo carga el decoder y nunca entrena.

## Limitaciones, riesgos y supervisión
- La arquitectura densa y el latente de 2D producen imágenes borrosas y poco diversas.
- Las clases con estructuras finas (sandalias, bolsos) fallan.
- Ambos modelos siguen mejorando en la última época: el early stopping no llega a activarse y más épocas podrían mejorar la reconstrucción.
- El análisis de vecinos por píxeles es una medida limitada.
- **Los datos sintéticos no garantizan privacidad, originalidad ni ausencia de sesgo.** Cualquier uso fuera del ámbito académico requiere revisión humana.
- Los modelos entrenados se versionan en Git (`models/autoencoder.keras` y `models/decoder.keras`), por lo que un despliegue desde el repositorio incluye el decoder que usa la aplicación sin reentrenar. Si se cambia el entrenamiento, hay que regenerarlos y volver a versionarlos.
