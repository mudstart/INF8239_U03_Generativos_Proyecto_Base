# Model Card · Autoencoder y VAE

## Modelos y versiones
- **Autoencoder (AE)**, guardado en `models/autoencoder.keras`, y **decoder del VAE**, en `models/decoder.keras`. El encoder del VAE no se guarda.
- Proyecto `inf8239-u03-generative` v0.1.0. Python 3.12, TensorFlow 2.21.0, Keras 3.15.1 y Streamlit 1.64.0.
- Entrenados con `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` y semilla 42. Una segunda ejecución reprodujo exactamente las mismas pérdidas.

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
| Entrenamiento | 54,000 | 60,000 |
| Validación | 6,000 (`validation_split=0.1`) | — (no tiene) |
| Prueba | 10,000 reservadas; 16 usadas en el panel (semilla 42) | igual |

- **Limitación:** no hay ninguna métrica numérica calculada sobre el conjunto de prueba completo.

## Arquitectura y espacio latente
| Modelo | Arquitectura | Latente | Parámetros |
|---|---|---|---|
| AE | 784 → Dense(16, ReLU) → Dense(784, sigmoid) | 16D determinista | 25,888 |
| Encoder del VAE | 784 → Dense(128, ReLU) → `z_mean`, `z_log_var` | 2D probabilístico | 100,996 |
| Decoder del VAE | 2 → Dense(128, ReLU) → Dense(784, sigmoid) | — | 101,520 |

- El VAE usa el truco de reparametrización `z = μ + exp(0.5·logσ²)·ε`, con una pérdida BCE sumada más la KL frente a N(0, I).
- Los ejes del latente **no tienen un significado garantizado**.
- El AE y el VAE **no son directamente comparables**: los latentes y las arquitecturas son distintos.

## Métricas de reconstrucción y generación
| Métrica | AE | VAE |
|---|---|---|
| BCE de entrenamiento (última época) | 0.3060 por píxel | 267.35 sumada (≈0.341 por píxel) |
| BCE de validación | **0.3071** (mejor punto, época 6) | — |
| KL (última época) | — | 6.26 (estable en torno a 6.2) |
| Sobreajuste | No: las curvas convergen | No evaluable (sin validación) |

- **Reconstrucción:** se conservan las siluetas y se pierden las texturas. Fallan las sandalias (pasan a botín o zapatilla) y los bolsos (pasan a rectángulos difusos o a algo parecido a un pulóver).
- **Generación:** de 16 muestras de N(0, I), la mayoría son prendas reconocibles pero borrosas. Dos son híbridos sin forma clara.
- **Interpolación:** continua (botín → camisa → pantalón), con fotogramas intermedios poco plausibles.

## Diversidad y vecinos cercanos
- **Diversidad baja:** unas 4 formas en total, y 5 de las 16 muestras son prácticamente la misma zapatilla. No aparecen bolsos, sandalias ni vestidos.
- **Sin colapso del latente:** la KL se mantiene en torno a 6.2, lejos de 0.
- **Vecinos más cercanos** (distancia euclidiana por píxeles sobre las 60,000 de entrenamiento): distancias de 1.63 a 3.55. No hay copias.
- **Limitación de la métrica:** favorece las imágenes oscuras. El mismo vecino (índice 11040) se repite 5 veces, y algunas clases no coinciden con la forma generada.

## Hardware, tiempo y tamaño
- **Hardware:** CPU AMD Ryzen 5 5600X (6 núcleos / 12 hilos), 32 GB de RAM, Windows 11 Pro nativo, sin GPU (TensorFlow ≥ 2.11 no admite GPU en Windows nativo).
- **Tiempo de entrenamiento:** AE ≈4.7 s y VAE ≈7.7–9.8 s, según la ejecución.
- **Tamaño en disco:** `autoencoder.keras` ≈339 KB y `decoder.keras` ≈426 KB.
- **Inferencia:** inmediata en CPU. La app solo carga el decoder y nunca entrena.

## Limitaciones, riesgos y supervisión
- La arquitectura densa y el latente de 2D producen imágenes borrosas y poco diversas.
- Las clases con estructuras finas (sandalias, bolsos) fallan.
- El VAE no tiene validación ni early stopping.
- El análisis de vecinos por píxeles es una medida limitada.
- **Los datos sintéticos no garantizan privacidad, originalidad ni ausencia de sesgo.** Cualquier uso fuera del ámbito académico requiere revisión humana.
- `models/*.keras` está en `.gitignore`: un despliegue desde Git no incluiría el decoder.
