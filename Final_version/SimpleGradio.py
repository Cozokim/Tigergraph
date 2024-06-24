# ----------------------------------------------------------------
# Pablo Alonso, Samuel Laso Y Gonzalo Bolado (LIS Data Solutions)
# ----------------------------------------------------------------

import gradio as gr
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.metrics import confusion_matrix

from io import BytesIO
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

import pickle

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from TrainingPipeline import TrainingPipeline
from PreprocessingPipeline import PreprocessingPipeline
from EDA import EDA, LIS_Score
from InferencePipeline import InferencePipeline
from FeatureSelector import WrapperMethods
from Plots import ResultsPlots

import logging
from logging.handlers import RotatingFileHandler
log_filename = 'Logs_accidentes.txt'
log_max_size_bytes = 5242880  # Tamaño máximo del archivo en bytes
log_backup_count = 5
# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Configuración del manejador de rotación de archivos
file_handler = RotatingFileHandler(log_filename, maxBytes=log_max_size_bytes, backupCount=log_backup_count)
file_handler.setLevel(logging.INFO)
# Formato del log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# Agregar el manejador al logger
logger.addHandler(file_handler)

class SimpleGradioMetro:
            
    def __init__(self):

        self.df = None
        self.target = None

        self.unique_ids_col = None
        self.unique_ids = None

        self.model_direction = 'Pickle/simplified_best_model'
        self.pipeline_direction = 'Pickle/simplified_preprocessing_pipeline'

        self.models_dict = {'NaiveBayes': {}, 'LogisticRegression': {}, 'KNN': {}, 'DecisionTrees': {}, 'RandomForest': {}, 'LightGBM': {}, 'XGBoost': {}, 'IForest': {}, 'GMM': {}, 'LOF': {}, 'NeuralNetwork': {}, 'Ensemble': {}, 'SVM': {}}

    def read_csv_path(self, input_files):

        self.df = pd.read_csv(input_files.name)
        target_column = sorted(list(self.df.columns))

        return gr.Dropdown(choices = target_column)

    def read_csv_test_path(self, input_files):

        self.df_test = pd.read_csv(input_files.name)
        columns_test = sorted(list(self.df_test.columns))

        return  gr.Dropdown(columns_test)

    def front_func(self):

        with gr.Blocks(css="Gradio/styles.css", title="SUPER CLASSIFIER") as demo:
            gr.Markdown("## ""![](file/Gradio/logo_lis.svg) SUPER CLASSIFIER""", elem_classes="cabecero")
            with gr.Tabs() as tabs:

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
                #                                    Introducción                                                 #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
                with gr.TabItem("Bienvenida",id=0):
                    gr.Markdown(

                        """
                        # Bienvenido al Super Clasificador!

                        Esta solución está pensada para solventar problemas de detección de anomalias y problemas de clasificación en base a históricos de datos. Algunos ejemplos de uso inclueyn:

                        1. **Detección de Anomalias**: Predecir si va a haber un accidente o no en una Obra.

                        2. **Clasificación binaria**: Predecir si una persona va a acudir a un evento o no.

                        3. **Clasificación multi-clase**: Predecir la calidad de un Producto: Bajo, Medio y Alto.

                        La versión a la que estás accediendo es una versión simplificada, especialmente pensada para usuarios que no tienen conocimiento de ciencia de datos, por lo que podrás realizar la predicción y simulaciones fácilmente. Si eres un usuario con experiencia en ciencia de datos, existe una versión del super clasificador mucho más completa en la que cada paso es totalmente configurable.

                        ## ¿Cómo utilizo el Super Clasificador?

                        ¡Es muy sencillo! Lo primero que necesitas es un conjunto de datos en formato .csv que contenga la variable a predecir y el resto de variables aparejadas. Por ejemplo, si quieres predecir la probabilidad de un accidente, necesitas un conjunto de datos que contenga algo similar a una columna “accidente” con valores “si/no” o "1/0".

                        Lo segundo que tienes que hacer es dividir ese conjunto de datos en dos diferentes (puedes cortar la mitad del contenido del .csv y llevarlo a otro .csv en blanco). El primer subconjunto generado lo utilizaremos para entrenar los modelos de predicción, y el segundo conjunto para validar las predicciones. ¡Por esto es importante que sean diferentes!

                        1.A continuación, sube el primer conjunto datos generados a la pestaña **Subir Datos**. Selecciona la variable objetivo y la métrica que te interesa predecir (Acurracy, Precision o Recall). Imagínate que estás tirando con arco a una diana:
                            
                        A. **Precision** mide si estás acertando exactamente donde quieres (el centro).
                        B. **Recall** verifica si acertaste en el centro en todas las oportunidades que tenías de darle.
                        C. **Acurracy** evalúa cuántas veces tus flechas dieron en cualquier parte de la diana comparado con los tiros totales, incluyendo los errores. Si disparaste 10 flechas y todas acertaron en la diana (sin importar si fue en el centro o no), y además no lanzaste ninguna fuera, entonces tu exactitud es perfecta.

                        2.Pulsa el botón **Subir Datos y Lanzar Entrenamiento** para que comience la magia. En esta pestaña haremos un procesamiento automático de los datos de entrenamiento y se utilizarán 12 potentes modelos de clasificación, nos quedaremos con el que mejor resultados da en base a tu problema. Te devolveremos, el mejor modelo con el resultado obtenido y la métrica LIS Score, la cual evalua la calidad del dato en base a la cantidad de información que falta, el balanceo de la clase objetivo y la cardinalidad de las variables. La métrica va de 0 a 100%. 

                        3.Por último, sube el otro conjunto de datos que generaste a la pestaña **inferencia** para realizar las predicciones de la variable objetivo.

                        4.Además de hacer predicciones con tus datos reales, tienes una pestaña de Gemelo Digital en la que puedes simular otros escenarios distintos y compararlos entre ellos (como qué pasaría si cambio el valor de esta variable). 

                        ¡Diviértete!

                        """
                        )
                    
                    button_intro =  gr.Button('Next Tab', elem_classes='button')
                    def change_tab_intro():
                            return gr.Tabs(selected=1)
                    
                    button_intro.click(change_tab_intro,None,tabs)
               
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
                #                              Upload Dataset & Train Models                                      #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   

                with gr.TabItem("Subir Datos",id=1):
                    title_EDA = gr.Markdown("## Exploración de Datos", elem_classes='cabecero')
                    
                    with gr.Row():
                        upload_file = gr.inputs.File(label='Subir el Dataset')

                        with gr.Column():
                            target_column_dropdown = gr.Dropdown(label = 'Seleccione la variable a predecir', choices = [],  interactive=True, allow_custom_value= True, elem_classes="dropdown")
                            metric_dropdown = gr.Dropdown(label = 'Seleccione la métrica', value = 'accuracy', choices = ['accuracy', 'recall', 'precision'], interactive=True, elem_classes="dropdown")
                        
                        upload_file.upload(self.read_csv_path, inputs = upload_file, outputs = [target_column_dropdown])

                    def upload_data(target_column_dropdown, metric_dropdown):
                        
                        try: 

                            self.target = target_column_dropdown
                            self.metric = metric_dropdown

                            lis_score = LIS_Score(df = self.df, target = self.target)
                            df_lis_score = round(lis_score.calculate_grades_score())

                            preproc = PreprocessingPipeline(df = self.df,
                                                            target = self.target,
                                                            threshold_cardinality = 75,
                                                            missing_data_method = 'stadistics',                                                              
                                                            preprocessing_filepath = self.pipeline_direction)

                            self.df_preprocessed = preproc.fit_transform()   

                            Trainer = TrainingPipeline(df = self.df_preprocessed,
                                                        target = self.target,
                                                        models_dict = self.models_dict,
                                                        n_splits = 3,
                                                        metric = self.metric,
                                                        training_filepath = self.model_direction)

                            _, _, _, best_model_text = Trainer.return_training_results()
                            training_text = 'El Entrenamiento ha terminado. ' + best_model_text  
                            update_EDA=(gr.Button(visible=True))
                  
                            logger.info('Se ha ejecutado correctamente la funcion guardar_valores. Tab: EDA')

                        except Exception as e:
                            logger.error('Ha fallado la funcion guardar_valores. Error: %s', e, exc_info=True)

                        return training_text, update_EDA, df_lis_score
                    
                    with gr.Row():    
                        button_train_models = gr.Button(value = 'Subir Datos y Lanzar Entrenamiento', elem_classes = 'button')

                    with gr.Row():
                        with gr.Column():
                            text_entrenamiento = gr.Textbox(label='Resultados Entrenamiento', value ='Lanzar el Entrenamiento 😀', elem_classes='textbox')
                        with gr.Column():
                            lis_score = gr.Textbox(label = 'LIS Score', elem_classes= 'textbox')

                    with gr.Row():
                        updatea = gr.Button("Next Tab", elem_classes="button", visible=False)
                        
                        button_train_models.click(upload_data, inputs = [target_column_dropdown, metric_dropdown], outputs = [text_entrenamiento,updatea, lis_score])
                      
                        def change_tab_EDA():
                                return gr.Tabs(selected=2)
                        updatea.click(change_tab_EDA,None,tabs)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
                #                                  Inferencia                                                     #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
          
                with gr.TabItem("Predicciones",id=2):
                   
                    with gr.Row():  

                        def boton_inferencia_params(unique_ids, probabilistic_results):

                            try:
                                self.unique_ids = None if unique_ids == 'None' else unique_ids

                                inference = InferencePipeline(training_filepath = self.model_direction, 
                                                            preprocessing_filepath = self.pipeline_direction, 
                                                                unique_ids = self.unique_ids)
                                
                                # Drop target column to prevent errors
                                if self.target in self.df_test.columns: 
                                    self.df_test = self.df_test.drop(columns=self.target)
                                
                                df_test = inference.inference_preprocessing(df_test = self.df_test)

                                results_proba = True if probabilistic_results == 'True' else False
                                self.predictions = inference.inference_predictions(df_test = df_test, results_proba = results_proba)                        

                                logger.info('Se ha ejecutado correctamente la funcion boton_inferencia_params. Tab: Inferencia')

                            except Exception as e:
                                logger.error('Ha fallado la funcion boton_inferencia_params. Error: %s', e, exc_info=True)

                            return gr.Dataframe(self.predictions, elem_classes='textbox', visible=True), update_inferencia
                    
                        def change_tab_inferencia():
                            return gr.Tabs(selected=4)
                
                        with gr.Column(scale=1):
                                    upload_test_files = gr.inputs.File(label='Subir dataset')
                                
                        with gr.Column(scale=1):
                                unique_ids = gr.Dropdown(label='Columna identificatoria', value = 'None', choices=[], multiselect=True, interactive=True, elem_classes='dropdown')
                                probabilistic_results = gr.Dropdown(label='Predicciones Probabilisticas (%)', choices=['True', 'False'], interactive=True, value='False',elem_classes='dropdown')
                                
                                upload_test_files.upload(self.read_csv_test_path, inputs = upload_test_files, outputs=[unique_ids])

                    with gr.Row():
                        with gr.Column(scale=1):
                            boton_inference = gr.Button("Lanzar Predicciones", elem_classes="button") 
                            inputs_boton_inference = [unique_ids, probabilistic_results]

                    with gr.Row():
                        with gr.Column(scale=1):

                            predictions_text = gr.Dataframe(label='Predicciones', elem_classes='textbox', show_label= True, visible=False)

                    with gr.Row():
                        update_inferencia = gr.Button("Next Tab", elem_classes="button", visible=False)
                    
                        boton_inference.click(boton_inferencia_params, inputs=inputs_boton_inference, outputs = [predictions_text, update_inferencia]) 
                        update_inferencia.click(change_tab_inferencia,None,tabs)
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
                #                                     GEMELO DIGITAL                                              #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

                with gr.TabItem("Gemelo Digital",id=5):  
                    title_gemelo_digital = gr.Markdown("## Gemelo Digital", elem_classes = "cabecero")
                
                    with gr.Row():
                        boton_gemelo = gr.Button(value = "Actualizar Columnas", elem_classes= 'button')

                    with gr.Row():
                        with gr.Column():

                            NUM_OBJECTS = 50

                            slider_list = []
                            for i in range(NUM_OBJECTS): # IMPORTANT: el dataset puede tener mas de 25 variables. 
                                slider = gr.Slider(visible = False, elem_classes = 'slider')
                                slider_list.append(slider)

                        with gr.Column():

                            dropdown_list = []
                            for i in range(NUM_OBJECTS): #IMPORTANT: el dataset puede tener mas de 25 variables. 
                                dropdown = gr.Dropdown(visible  = False, elem_classes='dropdown')
                                dropdown_list.append(dropdown)

                        def _show_sliders():
                             
                            try:
                                
                                def update_sliders(numerical_df, numerical_slider):

                                    # Updates sliders if the values are equal to not raise a math error.
                                    equal_values = {feature:{'min_value': int(numerical_df[feature].min()), 'max_value': int(numerical_df[feature].max())} for feature in self.numerical_col_names}
                                    equal_features = [key for key, value in equal_values.items() if value['min_value'] == value['max_value']]
                                    
                                    index_list = [index for index, value in enumerate(numerical_slider) if value['label'] in equal_features]
                                    for index, feature_dict in enumerate(numerical_slider):
                                        if index in index_list:
                                            feature_dict['minimum'] =- 1
                                            
                                    return numerical_slider
                                
                                with open(self.pipeline_direction + '.pkl', 'rb') as file:
                                    preprocessing_objects = pickle.load(file)
                                    self.target = preprocessing_objects['target']
 
                                bad_cols = [self.target]
 
                                numerical_df = self.df_test.select_dtypes(include ='number')
                                self.numerical_col_names = [col for col in numerical_df.columns if col not in bad_cols]
 
                                slider_list_num = [gr.Slider.update(label = f'{column}', visible = True, interactive = True, value = int(numerical_df[column].min()), minimum = int(numerical_df[column].min()), maximum = int(numerical_df[column].max())) for column in self.numerical_col_names]
                                slider_list_num = update_sliders(numerical_df = numerical_df, numerical_slider = slider_list_num)
                                
                                update_hide_num = [gr.Slider.update(visible=False, value="") for _ in range(NUM_OBJECTS - len(self.numerical_col_names))]
 
                                categorical_df = self.df_test.select_dtypes(include = 'object')
                                self.categorical_col_names = [col for col in categorical_df.columns if col != self.target]
 
                                dropdown_list_cat = [gr.Dropdown.update(label = f'{column}', visible = True, interactive = True, value = categorical_df[column].mode()[0], choices = categorical_df[column].unique()) for column in self.categorical_col_names]
                                update_hide_cat = [gr.Dropdown.update(visible = False, value="") for _ in range(NUM_OBJECTS - len(categorical_df.columns) + 1) ]
                                               
                                
                                logger.info('Se ha ejecutado correctamente la funcion show_sliders. Tab: Gemelo Digital')
 
                            except Exception as e:
                                logger.error('Ha fallado la funcion show_sliders. Error: %s', e, exc_info=True)
        
                            return [item for sublist in (slider_list_num + update_hide_num, dropdown_list_cat + update_hide_cat) for item in sublist]
                    
                        boton_gemelo.click(_show_sliders, outputs = slider_list  + dropdown_list)

                    with gr.Row():
                        with gr.Column(scale=1):
                            boton_gemelo_digital = gr.Button("Run", elem_classes="button")

                            def resultados_real_time(*inputs_btn_gemelo):
 
                                if 'resultados_df' not in globals():
                                    global resultados_df
                                    resultados_df = pd.DataFrame()
 
                                try:
 
                                    inputs_btn_gemelo = [col for col in inputs_btn_gemelo if col != ""]
                                    cols = [col for col in self.numerical_col_names + self.categorical_col_names if col != self.target]
                                    
                                    resultado_df = pd.DataFrame([inputs_btn_gemelo], columns=cols)
                                    x_df = pd.DataFrame([inputs_btn_gemelo], columns=cols)
                                    
                                    inference = InferencePipeline(self.model_direction, self.pipeline_direction, unique_ids=self.unique_ids_col)
                                    resultado_df = inference.inference_preprocessing(df_test=resultado_df)
                                                                                                                               
                                    predictions = inference.inference_predictions(df_test = resultado_df, results_proba= True)
                                    x_df = pd.concat([predictions, x_df],axis = 1)

                                    resultados_df = pd.concat([resultados_df, x_df], ignore_index = True)
                                    df = gr.Dataframe(elem_classes='textbox', visible=True, value=resultados_df)  

                                    results_plots = ResultsPlots(predictions = predictions)
                                    img1 = results_plots.plot_gemelo_digital_predictions()     

                                    image_predictions = gr.Image(img1, elem_classes='textbox', visible=True)                      
                                    logger.info('Se ha ejecutado correctamente la funcion resultados_real_time. Tab: Gemelo Digital')
                                    
                                except Exception as e:
                                    logger.error('Ha fallado la funcion resultados_real_time. Error: %s', e, exc_info=True)
                                
                                return df, image_predictions
                            
                    with gr.Row():
                        df = gr.Dataframe(visible=False, elem_classes= "textbox") 
                        image_predictions = gr.Image(visible = False, elem_classes='textbox')

                    inputs_btn_gemelo = slider_list + dropdown_list
                    boton_gemelo_digital.click(resultados_real_time, inputs=[*inputs_btn_gemelo], outputs = [df, image_predictions])     

            demo.launch(favicon_path = "Gradio/favicon.png")
                
if __name__ == "__main__":

    lis = SimpleGradioMetro()

    lis.front_func()
                         






   