import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
    #Extrae características de textura usando GLCM (Matriz de Co-ocurrencia de Niveles de Gris).

    # Calcular GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Extraer propiedades de la GLCM (contraste, energía, homogeneidad, correlación)
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    correlation = graycoprops(glcm, 'correlation')

    # Combinar todas las características en un vector
    features_glcm = np.hstack([contrast, energy, homogeneity, correlation])

    return features_glcm.flatten()

def extract_sift_descriptors(image):
    #Extrae descriptores locales usando SIFT para una imagen dada.
    sift = cv2.SIFT_create()  # Crear objeto SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Extraer descriptores
    return descriptors

def create_bof_model(image_paths, num_clusters=50):
    #Crea un modelo BoF (Bag of Features) a partir de un conjunto de imágenes.
    sift_descriptors = []

    # Extraer descriptores SIFT para todas las imágenes
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            descriptors = extract_sift_descriptors(img)
            if descriptors is not None:
                sift_descriptors.append(descriptors)

    # Concatenar todos los descriptores
    sift_descriptors = np.vstack(sift_descriptors)

    # Aplicar K-means para crear el vocabulario visual
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sift_descriptors)

    return kmeans

def extract_bof_features(image, kmeans_model, num_clusters=50):
    #Extrae las características BoF para una imagen dada usando el modelo de K-means.
    descriptors = extract_sift_descriptors(image)
    if descriptors is not None:
        # Asignar cada descriptor a su cluster correspondiente
        labels = kmeans_model.predict(descriptors)

        # Crear un histograma de las palabras visuales
        bof_features = np.bincount(labels, minlength=num_clusters)

        return bof_features
    return None

def extract_combined_features(image_path, kmeans_model, num_clusters=50):
    #Combina las características GLCM y BoF en un solo vector.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises

    # Extraer características GLCM
    glcm_features = extract_glcm_features(img)

    # Extraer características BoF
    bof_features = extract_bof_features(img, kmeans_model, num_clusters)

    # Concatenar las características GLCM y BoF en un solo vector
    if bof_features is not None:
        combined_features = np.hstack([glcm_features, bof_features])
    else:
        combined_features = glcm_features

    return combined_features