# Formato esperado de los datos de sueño

Archivo CSV con las siguientes columnas mínimas:

- user_id: identificador del usuario (string)
- date: fecha de la noche (YYYY-MM-DD)
- sleep_start: timestamp ISO (p.ej. 2025-10-01T23:45:00)
- sleep_end: timestamp ISO
- duration_min: duración del sueño en minutos (opcional — si no existe se calculará)
- sleep_quality: puntuación de calidad (0-1 o 0-100)
- rem_min, light_min, deep_min: minutos en cada etapa (opcional)
- awakenings: número de veces que despertó
- avg_heart_rate: frecuencia cardiaca media durante la noche (opcional)

Puedes generar datos sintéticos con el script `data/generate_synthetic_sleep.py`.