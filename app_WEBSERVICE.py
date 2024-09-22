# Importamos las bibliotecas necesarias
import uvicorn
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
import tempfile  # Biblioteca para crear archivos temporales
import shutil  # Biblioteca para copiar archivos
from pycaret.regression import predict_model

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "best_model.pkl"
with open(model_path, 'rb') as model_file:
    modelo = pickle.load(model_file)

# Cargar base de predicción en kaggle
prueba = pd.read_csv( "prueba_APP.csv",header = 0,sep=";",decimal=",")
prueba.drop(columns=['Address','price'], inplace=True)

## Datos de Entrada
dominio =  'yahoo'
Tec = 'PC'
Avg = 33.946241
Time_App = 10.983977
Time_Web = 37.951489
Length = 3.050713

# Endpoint para realizar la predicción
@app.post("/upload-excel")
def upload_excel(file: UploadFile = File(...)):
    try:
        # Crear un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            # Leer el archivo Excel usando pandas y almacenarlo en un DataFrame
            df = pd.read_excel(temp_file.name)
    
            # Realizar predicción
            predictions = predict_model(modelo, data=prueba)
            predictions["price"] = predictions["prediction_label"]
            prediction_label = list(predictions["price"])

            return {"predictions": prediction_label}

    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}


    # Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
