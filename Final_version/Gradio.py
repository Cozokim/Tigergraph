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

#TODO: add logs with try-except blocks

class BaseGradio:

    def __init__(self):

        self.columns = []
        self.target = None

        self.unique_ids_col = None
        self.unique_ids = None

        self.predictions = None
        self.deterministic_predictions = None

        self.df = pd.DataFrame()
        self.df_test = pd.DataFrame()

        self.labels = pd.DataFrame()
        self.best_features = None

        self.numerical_col_names = None
        self.categorical_col_names = None

        self.training_path = 'Pickle/training_pipeline'
        self.preprocessing_path = 'Pickle/preprocessing_pipeline'

        self.gemelo_digital_path = 'Pickle/gemelo_digital'

    def read_csv_path(self, input_files):

        self.df = pd.read_csv(input_files.name)

        target_column = sorted(list(self.df.columns))
        total_columns = ['None'] + sorted(list(self.df.columns))

        return  gr.Dropdown(choices = target_column), gr.Dropdown(choices = total_columns), gr.Dropdown(choices = total_columns), gr.Dropdown(choices = total_columns)

    def read_csv_test_path(self, input_files):

        self.df_test = pd.read_csv(input_files.name)
        columns_test = sorted(list(self.df_test.columns))

        return  gr.Dropdown(columns_test)
    
    def read_csv_labels_path(self, input_files):

        self.labels = pd.read_csv(input_files.name)
        return pd.DataFrame(self.labels) 
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
#                                 EDA : Exploratory Data Analysis                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    

    def eda_front(self):
        global upload_file
        global target_column_dropdown

        with gr.TabItem("EDA",id=0):
            title_EDA = gr.Markdown("## Exploración de Datos", elem_classes='cabecero')
            with gr.Row():

                upload_file = gr.inputs.File(label='Suba el Dataset')
                target_column_dropdown = gr.Dropdown(label='Seleccione la variable a predecir', choices = [],  interactive=True, allow_custom_value= True, elem_classes="dropdown")

            def plots_EDA(target_column_dropdown):
                
                self.target = target_column_dropdown

                eda = EDA(df = self.df, target = self.target)
                eda.calculate_results()

                recomendaciones = str(eda.recommendations())
                lis_score = LIS_Score(df = self.df, target = self.target)

                df_lis_score = round(lis_score.calculate_grades_score())
                plot_missing_data, plot_target_distribution = eda.eda_plots()

                button_update_EDA = (gr.Button(visible=True))
                return recomendaciones, df_lis_score, plot_missing_data, plot_target_distribution, button_update_EDA
            
            with gr.Row():
                eda_recomendaciones = gr.Textbox(label="Recomendaciones", lines=3, max_lines=6, elem_classes='textbox', visible=True)
                df_lis_score = gr.Number(label = "LIS Score", elem_classes="textbox", scale = 1, info = 'Métrica que evalúa la calidad de los datos utilizados en relación con varios criterios. Valor máximo: 100 Valor mínimo: 0')

            with gr.Row():
                button_EDA = gr.Button("Lanzar EDA", elem_classes="button")

            with gr.Row():
                plot_missing_data = gr.Image(elem_classes='textbox')
                plot_target_distribution = gr.Image(elem_classes='textbox')
    
            with gr.Row():
                button_update_EDA = gr.Button("Next Tab", elem_classes="button", visible=False)
                button_EDA.click(plots_EDA, inputs = [target_column_dropdown], outputs = [eda_recomendaciones, df_lis_score, plot_missing_data, plot_target_distribution, button_update_EDA])
                
                def change_tab_EDA():
                    return gr.Tabs(selected=1)
                
                button_update_EDA.click(change_tab_EDA,None,tabs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
    #                                 Pre-Procesamiento                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   

    def preproc_front(self):

        with gr.TabItem("Preprocesado",id=1):
            with gr.Row():
                
                with gr.Column(scale=1):
                    
                    title_missing_data = gr.Markdown("## Missing Data", elem_classes="cabecero")
                    na_threshold = gr.Slider(minimum=0, maximum = 100, label = "Threshold", value = 25, interactive= True, step = 1, elem_classes="slider")
                    missing_data_method = gr.Dropdown(label = "Método Imputar Missing Values", value="Valores Estadisticos", choices= ['ExtraTrees', 'Valores Estadisticos'], elem_classes= 'dropdown')
                
                    title_threshold = gr.Markdown("## Variables Cardinales", elem_classes = "cabecero")
                    threshold_cardinality = gr.Slider(minimum=0, maximum=100, label = "Cardinality Threshold", value = 75, interactive = True, step = 1, elem_classes='slider')

                    title_encoding = gr.Markdown("## Variables Categóricas", elem_classes="cabecero")
                    ordinal_variables = gr.Dropdown(label = "Variables Ordinales", choices = [], value = "None",  multiselect= True, interactive = True, elem_classes='dropdown') #
            
                with gr.Column(scale=1):

                    title_oversampling = gr.Markdown("## Oversampling", elem_classes="cabecero")
                    oversampling_method = gr.Dropdown(label="Método", choices = ["None", "Random", "Smote",'Smoten',], value= "None", elem_classes='dropdown')
                    oversampling_pct = gr.Slider(value=10, label= "Porcentaje variable objetivo (%)", step = 1, elem_classes= 'slider')

                    title_synthethic_data = gr.Markdown("## Datos Sintéticos con GANs", elem_classes="cabecero")
                    synthethic_samples =  gr.Number(label="Número de datos", elem_classes='textbox')
                    
                    upload_file.upload(self.read_csv_path, inputs = upload_file, outputs = [target_column_dropdown, ordinal_variables])

                    def launch_preprocessing(na_threshold ,missing_data_method , threshold_cardinality, oversampling_method,
                                                oversampling_pct, ordinal_variables, synthethic_samples):

                        ordinal_variables = [] if ordinal_variables == ['None'] else ordinal_variables
                        oversampling_method = None if oversampling_method == 'None' else oversampling_method

                        preproc = PreprocessingPipeline(df = self.df,
                                                        target = self.target,

                                                        threshold_cardinality = threshold_cardinality,
                                                        ordinal_features = ordinal_variables,

                                                        missing_data_method = missing_data_method,
                                                        missing_data_threshold= na_threshold,

                                                        oversampling_method = oversampling_method,
                                                        num_samples_pct =  oversampling_pct,

                                                        synthethic_samples= synthethic_samples,
                                                        preprocessing_filepath = self.preprocessing_path)
                        
                        self.df_preprocessed = preproc.fit_transform()

                        preproc_text = preproc.preprocessing_text()
                        preproc_text += '\n El dataset ya está listo 😁. \n'

                        button_update_preproc = (gr.Button(visible=True))
                        preproc_textbox = gr.Textbox(label="Resultados Preprocesamiento ", value = preproc_text, elem_classes='textbox')

                        return preproc_textbox, button_update_preproc
       
            with gr.Row():
                with gr.Column(scale = 1):

                    button_preproc = gr.Button("Lanzar Procesamiento de Datos", elem_classes= "button")
                    inputs_boton_preproc = [na_threshold, missing_data_method, threshold_cardinality, oversampling_method, oversampling_pct, ordinal_variables, synthethic_samples]
                
            with gr.Row():
                with gr.Column(scale = 1):
                    text_eda_recomendations = gr.Textbox(value = 'El dataset no esta listo 😔. Lanze el preprocesamiento. ' , elem_classes = 'textbox', label = 'Resultados PreProcesamiento') 

            with gr.Row():
                with gr.Column(scale=1):

                    def change_tab_preproc():
                        return gr.Tabs(selected=2)

                    button_update_preproc = gr.Button("Next Tab", elem_classes= "button", visible=False)
                    button_update_preproc.click(change_tab_preproc,None,tabs)
                    button_preproc.click(launch_preprocessing, inputs = inputs_boton_preproc, outputs = [text_eda_recomendations, button_update_preproc])    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
    #                                    Entrenamiento                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

    def entrenamiento_front(self):

        with gr.TabItem("Entrenamiento",id=2):
                with gr.Row():
                    with gr.Column(scale=1):

                        title_best_params = gr.Markdown("## Entrenamiento", elem_classes="cabecero")
                        choices_models = ['NaiveBayes', 'LogisticRegression','SVM','KNN', 'DecisionTrees', 'LightGBM', 'XGBoost', 'IForest', 'GMM', 'LOF', 'NeuralNetwork', 'Ensemble']

                        models_list = gr.Dropdown(label='Modelos', choices = choices_models, multiselect = True, interactive = True, elem_classes='dropdown')
                        n_splits_cv = gr.Slider(value=3, label="Número de splits", minimum = 2, maximum = 10, interactive = True, step =1, elem_classes='slider')

                        metric = gr.Dropdown(label = 'Métrica evaluar modelos', choices = ['accuracy', 'recall', 'precision'], value = 'accuracy', elem_classes='dropdown')

                        def func_training(models_list, n_splits_cv, metric, bool_best_params, n_trials, bool_best_features):  

                            models_dict = {model: {} for model in models_list}
                            bool_best_features = True if bool_best_features == 'True' else False
                            bool_best_params = True if bool_best_params == 'True' else False

                            Trainer = TrainingPipeline(
                                            df = self.df_preprocessed,
                                            target = self.target,

                                            models_dict = models_dict,
                                            n_splits = n_splits_cv,

                                            metric = metric,
                                            training_filepath =  self.training_path,
                                            use_best_features = bool_best_features,

                                            use_best_params = bool_best_params,
                                            n_trials =  n_trials)

                            best_features_text, model_params, training_metrics, best_model_text = Trainer.return_training_results()
                            self.best_features = best_features_text

                            button_update_training = (gr.Button(visible=True))                                    

                            return model_params, training_metrics, best_model_text, button_update_training, best_features_text

                        def change_tab_entrenamiento():
                                return gr.Tabs(selected=3)
                        
                    with gr.Column(scale=1):
                        title_best_params = gr.Markdown("## Optimización Parámetros", elem_classes="cabecero")
                        bool_best_params = gr.Dropdown(label = 'Entrenar con los mejores párametros', choices = ["True", "False"], interactive= True, value = 'False', elem_classes='dropdown')
                        n_trials = gr.Slider(label = "Número de iteraciones", minimum= 2, maximum=50, value = 10,step = 1,  elem_classes = 'dropdown')
                        bool_best_features = gr.Dropdown(label = "Entrenar con las mejores variables", choices = ["True", "False"], value = "False", elem_classes='dropdown', interactive = True)
                    
                with gr.Row():
                    with gr.Column(scale=1):

                        button_training = gr.Button("Lanzar Entrenamiento", elem_classes="button") 
                        inputs_training = [models_list, n_splits_cv, metric, bool_best_params, n_trials, bool_best_features]

                with gr.Row():
                    with gr.Column(scale=1):

                        training_results = gr.Markdown("## Resultados del Entrenamiento", elem_classes="cabecero")
                        model_params = gr.Textbox(label="Parámetros de los modelos", elem_classes="textbox")

                        training_metrics = gr.Textbox(label= "Métricas en el Entrenamiento", elem_classes="textbox")
                        best_model_text = gr.Textbox(label="Mejor modelo y resultados", elem_classes= "textbox")

                        best_features_text = gr.Textbox(label = "Variables Utilizadas", elem_classes = "textbox")

                with gr.Row():
                    button_update_training = gr.Button("Next Tab", elem_classes="button", visible=False)
                    
                    button_training.click(func_training, inputs = inputs_training, outputs = [model_params, training_metrics, best_model_text, button_update_training, best_features_text])
                    button_update_training.click(change_tab_entrenamiento,None,tabs)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
    #                                  Inferencia                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

    def inferencia_front(self):

         with gr.TabItem("Inferencia",id=3):
            title_inferencia = gr.Markdown("## Inferencia", elem_classes="cabecero")

            with gr.Row():  

                def boton_inferencia_params(unique_ids, probabilistic_results):

                    self.unique_ids = None if unique_ids == 'None' else unique_ids
                    results_proba = True if probabilistic_results == 'True' else False

                    inference = InferencePipeline(preprocessing_filepath = self.preprocessing_path, 
                                                  training_filepath = self.training_path,
                                                  unique_ids = self.unique_ids)
                    
                    df_test = inference.inference_preprocessing(df_test = self.df_test)
                    self.predictions = inference.inference_predictions(df_test = df_test, results_proba = results_proba)                        

                    deterministic_predictions = inference.inference_predictions(df_test = df_test, results_proba = False) 
                    self.deterministic_predictions = deterministic_predictions.drop(columns = self.unique_ids) if self.unique_ids else deterministic_predictions

                    return gr.Dataframe(self.predictions, elem_classes='cabecero', visible=True), update_inferencia
            
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
    #                                  Resultados                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

    def resultados_front(self):

        with gr.TabItem("Resultados",id=4):
            title_results = gr.Markdown("## Resultados", elem_classes= "cabecero")

            with gr.Row():

                def resultados_plots():
                    
                    results_plots = ResultsPlots(predictions = self.deterministic_predictions)
                    plot_predictions = results_plots.plot_results()
                    return plot_predictions
            
                plots_predictions = gr.Image(height = 600, width = 1000, elem_classes = 'textbox')

            with gr.Row(): 

                button_plots_resultados = gr.Button('Lanzar los Plots', elem_classes='button')
                button_plots_resultados.click(resultados_plots, outputs = [plots_predictions])
                        
            with gr.Row():
                title_confusion_matrix= gr.Markdown("## Confusion Matrix", elem_classes = "cabecero")

            with gr.Row():

                def confusion_matrix_resultados():

                    results_plots = ResultsPlots(predictions = self.deterministic_predictions)
                    conf_matrix, results_inference = results_plots.plot_confusion_matrix(df_labels = self.labels)
                
                    return conf_matrix, results_inference

                upload_labels = gr.inputs.File(label = 'Subir Categorías de las Predicciones')

                upload_labels.upload(self.read_csv_labels_path, inputs = upload_labels)
                plot_confusion_matrix = gr.Image(height = 600, width = 800, elem_classes = "textbox")

            with gr.Row():
                results_inference = gr.Textbox(label='Métricas en Inferencia',  elem_classes= "textbox")

            with gr.Row():
                button_metrics = gr.Button('Lanzar Resultados',elem_classes = 'button')
                button_metrics.click(confusion_matrix_resultados, outputs = [plot_confusion_matrix, results_inference])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
    #                                     GEMELO DIGITAL                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def gemelo_digital_front(self):

        with gr.TabItem("Gemelo Digital",id=5):  
            title_gemelo_digital = gr.Markdown("## Gemelo Digital", elem_classes = "cabecero")
        
            with gr.Row():
                button_gemelo = gr.Button(value = "Actualizar Columnas", elem_classes= 'button')

            with gr.Row():
                with gr.Column():

                    NUM_OBJECTS = 50  # IMPORTANT: el dataset puede tener mas de 50 variables ó númericas o categóricas. 

                    slider_list = []
                    for i in range(NUM_OBJECTS): 
                        slider = gr.Slider(visible = False, elem_classes = 'slider')
                        slider_list.append(slider)

                with gr.Column():

                    dropdown_list = []
                    for i in range(NUM_OBJECTS):
                        dropdown = gr.Dropdown(visible  = False, elem_classes='dropdown')
                        dropdown_list.append(dropdown)

                def show_sliders():

                    with open(self.preprocessing_path + '.pkl', 'rb') as file:
                        preprocessing_objects = pickle.load(file)

                        self.target = preprocessing_objects['target']
                        self.numerical_col_names = preprocessing_objects['numerical_col_names']
                        self.categorical_col_names = preprocessing_objects['categorical_col_names']
                        
                    def update_sliders(numerical_df, numerical_slider):

                        # Updates sliders if the values are equal to not raise a math error.
                        equal_values = {feature:{'min_value': int(numerical_df[feature].min()), 'max_value': int(numerical_df[feature].max())} for feature in self.numerical_col_names}
                        equal_features = [key for key, value in equal_values.items() if value['min_value'] == value['max_value']]
                        
                        index_list = [index for index, value in enumerate(numerical_slider) if value['label'] in equal_features]
                        for index, feature_dict in enumerate(numerical_slider):
                            if index in index_list: feature_dict['minimum'] =- 1
                                
                        return numerical_slider
                    
                    if self.df_test is None or self.df_test.empty:

                        with open(self.gemelo_digital_path + '.pkl', 'rb') as file:
                            gemelo_objects = pickle.load(file)                           
                        return gemelo_objects
                    
                    else:

                        numerical_df = self.df_test.select_dtypes(include ='number')
                        slider_list_num = [gr.Slider.update(label = f'{column}', visible = True, interactive = True, value = int(numerical_df[column].min()), minimum = int(numerical_df[column].min()), maximum = int(numerical_df[column].max())) for column in self.numerical_col_names]
                        slider_list_num = update_sliders(numerical_df = numerical_df, numerical_slider = slider_list_num)
                        
                        update_hide_num = [gr.Slider.update(visible=False, value="") for _ in range(NUM_OBJECTS - len(self.numerical_col_names))]

                        categorical_df = self.df_test.select_dtypes(include = 'object')

                        dropdown_list_cat= [gr.Dropdown.update(label = f'{column}', visible = True, interactive = True, value = categorical_df[column].mode()[0], choices = categorical_df[column].unique()) for column in self.categorical_col_names]
                        update_hide_cat = [gr.Dropdown.update(visible = False, value="") for _ in range(NUM_OBJECTS - len(categorical_df.columns) + 1) ]

                        gradio_slider_dropdown = [item for sublist in (slider_list_num + update_hide_num, dropdown_list_cat + update_hide_cat) for item in sublist]
                        with open(self.gemelo_digital_path + '.pkl', 'wb') as file:
                            pickle.dump(gradio_slider_dropdown, file)
                    
                        return gradio_slider_dropdown
            
                button_gemelo.click(show_sliders, outputs = slider_list  + dropdown_list)

            with gr.Row():
                with gr.Column(scale=1):
                    button_gemelo_predictions = gr.Button("Lanzar Predicciones", elem_classes="button")
                    title_gemelo_predictions = gr.Markdown(value = '## Escenarios Generados', elem_classes= 'cabecero', visible = False)

                    def predictions_gemelo(*inputs_btn_gemelo):

                        if 'df_total_gemelo' not in globals():
                            global df_total_gemelo
                            df_total_gemelo = pd.DataFrame()

                        inputs_btn_gemelo = [col for col in inputs_btn_gemelo if col != ""]
                        cols = [col for col in self.numerical_col_names + self.categorical_col_names if col != self.target]

                        df_gemelo = pd.DataFrame([inputs_btn_gemelo], columns=cols)
                        df_gemelo_raw = pd.DataFrame([inputs_btn_gemelo], columns=cols)

                        inference = InferencePipeline(self.training_path, self.preprocessing_path, unique_ids=self.unique_ids_col)
                        df_gemelo = inference.inference_preprocessing(df_test=df_gemelo)

                        predictions = inference.inference_predictions(df_test=df_gemelo, results_proba = True)
                        df_gemelo_raw = pd.concat([predictions, df_gemelo_raw], axis=1)

                        df_total_gemelo = pd.concat([df_total_gemelo, df_gemelo_raw], ignore_index=True)
                        df = gr.Dataframe(elem_classes='dropdown', visible=True, value=df_total_gemelo)                                

                        return df , gr.Markdown(visible=True)
                    
            with gr.Row():
                predictions_gemelo_digital = gr.Dataframe(visible=False) 

            inputs_btn_gemelo = slider_list + dropdown_list
            button_gemelo_predictions.click(predictions_gemelo, inputs = [*inputs_btn_gemelo], outputs = [predictions_gemelo_digital, title_gemelo_predictions])      

class GradioMetro(BaseGradio):

    def __init__(self):
        super().__init__()

    def front_func(self):
        global tabs

        with gr.Blocks(css="Gradio/styles.css", title= "SUPER CLASSIFIER") as demo:
            gr.Markdown("## ""![](file/Gradio/logo_lis.svg) SUPER CLASSIFIER""", elem_classes= "cabecero")
            with gr.Tabs() as tabs:

                self.eda_front()
                self.preproc_front()
                
                self.entrenamiento_front()
                self.inferencia_front()

                self.resultados_front()
                self.gemelo_digital_front()
          
            demo.launch(favicon_path = "Gradio/favicon.png")
                
if __name__ == "__main__":

    lis = GradioMetro()
    
    lis.front_func()
                         






   
