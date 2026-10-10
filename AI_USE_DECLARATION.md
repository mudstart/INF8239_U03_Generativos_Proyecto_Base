# Declaración de uso de inteligencia artificial

## Herramientas utilizadas
- **Claude ** (aplicación de escritorio de Anthropic), modelo **Claude Opus 5.5**. Se usó como herramienta de apoyo durante la realización del laboratorio, siempre bajo la dirección del estudiante.
- No se usaron otras herramientas de IA en este laboratorio.

## Propósito de cada uso
La herramienta actuó como soporte en las siguientes tareas, cuya planificación, decisión y validación correspondieron al estudiante:

| Tarea del estudiante | Apoyo brindado por la herramienta |
|---|---|
| Comprender el proyecto base | Resumen de la estructura, el flujo de entrenamiento y los entregables pendientes |
| Interpretar las secciones 1 a 8 del manual | Apoyo para relacionar cada concepto con el código y para analizar los resultados (curvas del AE y del VAE, panel, interpolación) |
| Implementar los paneles no seleccionados (sección 7) | Asistencia en la modificación de `scripts/train_ae_vae.py`: panel de 16 ejemplos de prueba elegidos con semilla 42 y con etiquetas de clase, panel de vecinos más cercanos (`reports/nearest_neighbors.png`) y métricas ampliadas en `training_metrics.json` |
| Verificar la ejecución | Apoyo para volver a entrenar con los parámetros del manual, ejecutar las pruebas (`pytest`, 4 superadas) y `ruff` |
| Elaborar la documentación | Borradores del cierre interpretativo (`README.md`), de `MODEL_CARD.md` y de esta declaración, revisados y aprobados por el estudiante |
| Validar el VAE | Incorporación de un `test_step` al VAE y de la misma partición de validación (10 %) con early stopping que usa el AE, para poder evaluar el sobreajuste de ambos modelos. Como el VAE pasó a entrenarse con 54,000 imágenes, se recalcularon y actualizaron todas sus cifras en la documentación |
| Reforzar la evaluación y las pruebas | Incorporación de SSIM (`tf.image.ssim`) como métrica perceptual de fidelidad, complementaria a las métricas por píxel, y de `tests/test_training.py`, que ejecuta el entrenamiento completo con pocos datos, comprueba que se generan todos los artefactos y que dos ejecuciones dan resultados idénticos |

## Prompts o decisiones relevantes
- El laboratorio se abordó de forma incremental, una sección del manual a la vez. Para cada una, el asistente planteaba una propuesta y el estudiante determinaba si se ejecutaba, se ajustaba o se descartaba.
- La gestión del repositorio (commits y publicación de cambios) se reservó exclusivamente al estudiante; el asistente no realizó ninguna operación de control de versiones.
- Para la sección 7, la búsqueda de vecinos más cercanos se integró en el script de entrenamiento en lugar de crear un script independiente, ya que el encoder del VAE no se conserva tras el entrenamiento.
- La ubicación de cada entregable fue definida por el estudiante (por ejemplo, el cierre interpretativo en `README.md`), quien además revisó cada documento antes de incorporarlo al proyecto.

## Elementos verificados por el equipo
- Instalación del entorno y `check_runtime.py` ejecutados por el estudiante en su terminal.
- Entrenamiento completo ejecutado por el estudiante. Las pérdidas de su log coinciden con las de la ejecución de verificación, lo que confirma que el resultado es reproducible.
- `requirements-cloud.txt` generado por el estudiante con el comando del manual.
- Ejecución local de la aplicación de Streamlit: verificada por el estudiante. La app abre en `localhost:8501`, carga el decoder y genera imágenes desde el espacio latente.
- Revisión de los paneles, las cifras y las interpretaciones del README y del Model Card: cada documento se presentó primero como propuesta y el estudiante lo revisó y aprobó antes de incorporarlo. Las cifras proceden del log de entrenamiento del estudiante y de `reports/training_metrics.json`.
- Validación final del proyecto frente a las 10 secciones del manual U03.LAB10 y la rúbrica del Ejercicio 06, solicitada por el estudiante.
- Notebook ejecutado de principio a fin sin errores; reprodujo las mismas pérdidas que las ejecuciones desde la terminal (AE: val_loss 0.3071; VAE: val_loss 276.55, reconstrucción 268.10, KL 6.22).
- Suite de pruebas: 10 pruebas superadas (datos, modelos, métricas de evaluación, generación de artefactos y reproducibilidad del entrenamiento) y `ruff` sin avisos.

## Cambios realizados sobre resultados generados
- Los aportes de la herramienta se incorporaron después de que el estudiante los revisara. El estudiante orientó, acotó y, en el caso de esta declaración, editó directamente esos aportes:
  - Decidió que el cierre interpretativo se ubicara en `README.md`.
  - Solicitó incorporar al Model Card los datos reales de hardware de su equipo.
  - Aportó la salida de consola del entrenamiento y la captura de la aplicación como evidencia.
- **Corrección de reproducibilidad:** al ejecutar el notebook dos veces en el mismo kernel de VS Code, el AE obtuvo val_loss 0.3093 en lugar de 0.3071, porque las semillas solo se fijaban al importar el script. Se movieron al inicio de `main()` (`tf.keras.utils.set_random_seed`) y se activó `tf.config.experimental.enable_op_determinism()`; dos ejecuciones consecutivas en el mismo proceso dieron resultados idénticos a los documentados. El notebook además recarga el script (`importlib.reload`) para usar siempre su versión actual.
- **Corrección derivada de la evidencia:** la expectativa inicial era que las muestras generadas estarían más lejos del entrenamiento que las imágenes reales. El notebook mostró lo contrario (mediana 2.23 frente a 3.72), por lo que la interpretación se corrigió: la cercanía se explica por la borrosidad de las muestras y la naturaleza de la distancia por píxeles, no por copia.

## Responsabilidad asumida
El estudiante es el autor del trabajo y asume la responsabilidad total del contenido entregado. La herramienta de IA se utilizó exclusivamente como soporte para el análisis, el código y la redacción, sin sustituir su criterio ni sus decisiones. Las interpretaciones, cifras y conclusiones fueron revisadas frente a la evidencia del proyecto (log de entrenamiento, paneles y métricas). Las limitaciones declaradas en el Model Card, en especial que los datos sintéticos no garantizan privacidad, se asumen como parte del trabajo.
