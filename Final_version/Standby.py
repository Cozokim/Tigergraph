# ----------------------------------------------------------------
# Pablo Alonso, Samuel Laso Y Gonzalo Bolado (LIS Data Solutions)
# ----------------------------------------------------------------

 def plot_kpi(predictions):
 
predictions=round(predictions*100,2)
# Crear la figura y los ejes
fig, ax = plt.subplots()
# Dibujar el número grande del KPI
ax.text(0.5, 0.7, 'La probabilidad de accidente es:', fontsize=20, ha='center', va='center')
ax.text(0.5, 0.4, str(predictions) + '%', fontsize=100, ha='center', va='center')
# Eliminar los ejes
ax.axis('off')
# Mostrar la visualización
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
return img

def plot_kpi2(numero_variables):

# Redondear el número a dos decimales
numero_variables = round(numero_variables, 2)                                
# Crear la figura y los ejes
fig, ax = plt.subplots()                                
# Dibujar el texto descriptivo
ax.text(0.5, 0.7, 'Número de variables utilizadas:', fontsize=20, ha='center', va='center')                                
# Dibujar el número
ax.text(0.5, 0.4, str(numero_variables), fontsize=100, ha='center', va='center')                               
# Eliminar los ejes
ax.axis('off')                               
# Mostrar la visualización
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
return img

def plot_kpi3(texto):
    
# Crear la figura y los ejes
fig, ax = plt.subplots()

# Dibujar el texto descriptivo en la primera línea
ax.text(0.5, 0.7, 'Variable mas importante:', fontsize=20, ha='center', va='center')

# Dibujar el valor en la segunda línea
ax.text(0.5, 0.4, texto, fontsize=30, ha='center', va='center')
    
# Eliminar los ejes
ax.axis('off')

# Mostrar la visualización
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
return img
                        
def obtener_caracteristica_mas_importante(df, target_column):

df_sin_columna = df.drop(columns=[target_column])
columna_sola = df[target_column] 

# Entrenar un modelo de árbol de decisión
modelo = DecisionTreeClassifier()
modelo.fit(X=df_sin_columna, y=columna_sola)
# Obtener la importancia de las características
importancias = modelo.feature_importances_
indice_max_importancia = np.argmax(importancias)
caracteristica_mas_importante = df_sin_columna.columns[indice_max_importancia]

return caracteristica_mas_importante                           
