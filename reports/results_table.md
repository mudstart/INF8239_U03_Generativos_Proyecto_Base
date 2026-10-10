| Métrica | AE | VAE |
|---|---|---|
| BCE/píxel · prueba | 0.3068 | 0.3417 |
| MSE/píxel · prueba | 0.0203 | 0.0333 |
| SSIM · prueba (1 = idéntica) | 0.5363 | 0.4206 |
| Mejor val_loss (época) | 0.3071 BCE/píxel (6) | 276.55 BCE sumada + KL (8) |
| KL en validación (mejor época) | — | 6.12 |
| Genera desde N(0, I) | No | Sí |
| Mediana dist. vecino (muestras / prueba real) | — | 2.23 / 3.72 |
| Razón de diversidad (generadas / reales) | — | 0.53 |
| Parámetros | 25,888 | 202,516 |
| Tiempo de entrenamiento CPU (s) | 5.3 | 9.2 |
