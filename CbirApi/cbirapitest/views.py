from rest_framework.views import APIView
from rest_framework.response import Response
from .utils import extract_combined_features
from rest_framework.parsers import MultiPartParser, FormParser
from joblib import load
from rest_framework import status
import pandas as pd
import os

# Cargar el modelo KMeans guardado
model_path = os.path.join(os.path.dirname(__file__), 'kmeans_model.joblib')
kmeans_model = load(model_path)
print("Modelo cargado exitosamente.")

knn_path = os.path.join(os.path.dirname(__file__), 'knn_model.joblib')
knn = load(knn_path)
print("Modelo KNN cargado exitosamente.")

labels_path = os.path.join(os.path.dirname(__file__), 'labelindex.joblib')
labels = load(labels_path)
print("Etiquetas cargadas exitosamente.")



class SimpleAPIView(APIView):
    def get(self, request):
        # Datos de ejemplo para devolver
        data = {
            "message": "¡Hola! Esta es una API sin modelos.",
            "status": "success",
            "data": [1, 2, 3, 4, 5],
        }
        return Response(data)

class ImageFeatureExtractionView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Para manejar archivos

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')  # Obtener la imagen del formulario

        if not file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Guardar temporalmente la imagen en disco
        with open('temp_image.jpg', 'wb') as temp_file:
            for chunk in file.chunks():
                temp_file.write(chunk)
        

        # Procesar la imagen
        features = extract_combined_features('temp_image.jpg', kmeans_model, num_clusters=50)
        distances, indices = knn.kneighbors([features])

        # Crear la respuesta con las URLs de los índices encontrados
        result_urls = []
        for i in indices[0]:  # Iterar sobre los índices y obtener las URLs
            url = labels.iloc[i, 0]  # Obtener la URL de la columna 0
            result_urls.append(url)

        # Devolver las URLs en formato JSON
        return Response({"matched_images": result_urls}, status=status.HTTP_200_OK)

