# Mantenimiento predictivo de una máquina CNC

Proyecto final de Data Analytics en Gestión Industrial, Maestría en Gestión Industrial, Universidad de Ibagué.

Análisis del dataset AI4I 2020 Predictive Maintenance (UCI Machine Learning Repository, licencia CC BY 4.0) siguiendo cinco fases: contexto y selección del dataset, ETL, análisis exploratorio, modelado analítico (clustering, clasificación, PCA y LDA) y dashboard con plan de acción.

## Ver el dashboard

Dashboard en línea: 
Dashboard en local: http://localhost:8501

No requiere instalar nada; se abre desde el navegador.

## Archivos

| Archivo | Contenido |
|---|---|
| Proyecto.ipynb | Notebook con las cinco fases del proyecto |
| datast.cvs | Datos originales (AI4I 2020) |
| ai4i_datawarehouse.xlsx | Data Warehouse generado en el ETL |
| dashboard.py | Dashboard en Streamlit |
| dashboard_datos/ | Datos y resultados que exporta el notebook para el dashboard |
| requirements.txt | Librerías necesarias |
| Reporte_Proyecto.docx | Reporte final |

## Ejecutar el dashboard en tu computador

Requisito: Python 3.12.

### 1. Descargar el repositorio

Sin Git: en esta página, botón verde **Code → Download ZIP**, y descomprimir.

Con Git:

```
git clone https://github.com/FateZr/mantenimiento-predictivo-cnc.git
```

### 2. Entrar a la carpeta y crear el entorno virtual

Windows (PowerShell):

```
cd mantenimiento-predictivo-cnc
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac o Linux:

```
cd mantenimiento-predictivo-cnc
python3.12 -m venv .venv
source .venv/bin/activate
```

Si PowerShell muestra un error de "ejecución de scripts deshabilitada", ejecutar una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y volver a activar el entorno.

### 3. Instalar las librerías

```
pip install -r requirements.txt
```

### 4. Abrir el dashboard

```
streamlit run dashboard.py
```

Se abre en el navegador en http://localhost:8501. Para cerrarlo, presionar Ctrl + C en la terminal.

No es necesario ejecutar el notebook antes: la carpeta dashboard_datos/ ya contiene los resultados.

## Ejecutar el notebook

Con el entorno activado, abrir Proyecto.ipynb en VS Code o Jupyter, elegir el kernel .venv y ejecutar todas las celdas. El notebook descarga los datos, genera ai4i_datawarehouse.xlsx y actualiza la carpeta dashboard_datos/.

## Fuente de datos

Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. Licencia CC BY 4.0.
