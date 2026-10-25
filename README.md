# Dashboard · Pronóstico mensual de sismos por celda en Colombia

Versión mínima para desplegar en Render (solo lo que necesita el dashboard). El repositorio completo, con el libro,
las pruebas y los experimentos, se publica aparte en GitHub.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 120 --preload`
- Variable de entorno: `PYTHON_VERSION = 3.9.25`

Local: `pip install -r requirements.txt` y `python app.py` (http://127.0.0.1:8050).

Autores: Juan David Polo y Gabriella Pallares · Machine Learning, Universidad del Norte.
