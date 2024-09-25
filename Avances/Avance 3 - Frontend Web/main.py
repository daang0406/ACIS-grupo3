import streamlit as st
import numpy as np
import pydicom
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Clase para cargar y procesar imágenes DICOM
class DicomProcessor:
    def __init__(self):
        self.dicom_data = None

    def cargar_dicom(self, uploaded_file):
        self.dicom_data = pydicom.dcmread(uploaded_file)
        st.write("Información del archivo DICOM cargado:")
        st.write(self.dicom_data)

    def obtener_imagen(self):
        if self.dicom_data:
            imagen_array = self.dicom_data.pixel_array
            return imagen_array
        else:
            st.warning("No se ha cargado ningún archivo DICOM.")
            return None

    def mostrar_imagen(self):
        imagen = self.obtener_imagen()
        if imagen is not None:
            st.image(imagen, caption="Imagen DICOM", use_column_width=True)

# Clase para manejar el modelo de IA entrenado
class IA_Modelo:
    def __init__(self, modelo_path):
        self.modelo = self.cargar_modelo(modelo_path)

    def cargar_modelo(self, modelo_path):
        with open(modelo_path, 'rb') as file:
            modelo = pickle.load(file)
        st.success("Modelo IA cargado correctamente.")
        return modelo

    def predecir(self, imagen):
        if imagen is not None:
            # Aquí agregamos cualquier procesamiento adicional que sea necesario para el modelo.
            # Por simplicidad, asumimos que el modelo puede trabajar con el formato actual de la imagen.
            prediccion = self.modelo.predict([imagen.flatten()])  # Ejemplo simple, ajustar según el modelo real
            return prediccion
        else:
            st.warning("No se ha proporcionado una imagen válida para predecir.")
            return None

# Función principal que define la estructura de la aplicación con múltiples páginas
def main():
    # Título de la aplicación
    st.title("Aplicación de Procesamiento DICOM con IA")

    # Crear un menú de opciones para navegar entre las páginas
    with st.sidebar:
        menu_seleccionado = option_menu("Menú", ["Página 1: Cargar y Predecir DICOM", "Página 2: Personalizar"], 
                                        icons=["cloud-upload", "edit"],
                                        menu_icon="cast", default_index=0)

    # Crear instancia del procesador DICOM
    dicom_processor = DicomProcessor()

    # Página 1: Cargar y mostrar imagen DICOM
    if menu_seleccionado == "Página 1: Cargar y Predecir DICOM":
        st.header("Cargar Imagen DICOM y Predicción")

        uploaded_file = st.file_uploader("Elige un archivo DICOM", type=["dcm"])
        if uploaded_file is not None:
            dicom_processor.cargar_dicom(uploaded_file)
            dicom_processor.mostrar_imagen()

            # Cargar el modelo y hacer predicción
            modelo_ia = IA_Modelo("modelo_entrenado.pkl")
            imagen = dicom_processor.obtener_imagen()
            
            if st.button("Realizar Predicción"):
                prediccion = modelo_ia.predecir(imagen)
                if prediccion is not None:
                    st.write(f"Resultado de la predicción: {prediccion}")

    # Página 2: Personalizar (vacía)
    elif menu_seleccionado == "Página 2: Personalizar":
        st.header("Esta página está vacía. ¡Personalízala como prefieras!")

if __name__ == "__main__":
    main()
