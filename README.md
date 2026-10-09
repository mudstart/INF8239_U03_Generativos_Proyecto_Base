# INF-8239 · Unidad 03 · Modelos generativos

Autor académico: Edwin Ramón José Nolasco

## CPU
```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
```

## GPU
Solo para WSL2/Linux con NVIDIA correctamente configurada:
```bash
uv sync --extra gpu
```

## Aplicación
```bash
uv run streamlit run app/streamlit_app.py
```

Complete `MODEL_CARD.md` y `AI_USE_DECLARATION.md`. No afirme privacidad por el solo hecho de generar datos sintéticos.

## Cierre interpretativo

Evidencia: `reports/ae_vae_panel.png` (16 ejemplos de prueba elegidos con semilla 42), `reports/nearest_neighbors.png`, `reports/interpolation.png` y `reports/training_metrics.json`, obtenidos con `--epochs-ae 6 --epochs-vae 8`.

**Resultado de reconstrucción:** el AE (latente 16D) alcanza una BCE de validación de 0.3071 por píxel, sin sobreajuste: entrenamiento y validación descienden juntos durante las 6 épocas. El VAE (latente 2D) termina con una BCE sumada de 267.35 (≈0.341 por píxel, en entrenamiento). Ambos conservan la silueta, pero pierden texturas y detalles. Las sandalias y los bolsos fallan en los dos modelos. Las cifras no son directamente comparables porque los latentes y las arquitecturas difieren.

**Resultado de generación:** de las 16 muestras tomadas de N(0, I), la mayoría son prendas reconocibles pero borrosas. Dos son híbridos sin forma clara y no se genera ningún bolso, sandalia ni vestido reconocible.

**Evidencia de diversidad:** es baja. Las muestras se reducen a unas cuatro formas (zapatilla, prenda superior, pantalón y botín), y 5 de las 16 son prácticamente la misma zapatilla. Los vecinos más cercanos del entrenamiento están a distancias de 1.63 a 3.55, así que no hay copias. Sin embargo, la distancia por píxeles favorece las imágenes oscuras: el mismo vecino se repite 5 veces.

**Comportamiento del espacio latente:** la KL se estabiliza en torno a 6.2, sin colapso. La interpolación es continua (botín → camisa → pantalón), aunque los puntos intermedios son híbridos poco plausibles. Las prendas superiores (camiseta, camisa, pulóver y abrigo) se agrupan en una misma región. Los ejes no tienen un significado garantizado.

**Costo computacional:** entrenamiento en CPU (Windows nativo, TensorFlow 2.21). El AE tarda ≈4.7 s y el VAE ≈7.7–9.8 s. El AE tiene 25,888 parámetros y el decoder 101,520. Los archivos ocupan ≈339 KB (AE) y ≈426 KB (decoder). La inferencia en la aplicación es inmediata.

**Riesgo o limitación:** la arquitectura densa produce imágenes borrosas, la diversidad es limitada y las clases con estructuras finas fallan. El VAE se entrena sin validación y el análisis de vecinos por píxeles es una medida limitada. Una muestra sintética no es privada, original ni libre de sesgo por el solo hecho de ser sintética.

**Decisión sobre uso previsto:** apto para una demostración académica del espacio latente de un VAE. No apto para generar imágenes realistas, para aumentar datos de entrenamiento ni para sustituir datos reales por motivos de privacidad.
