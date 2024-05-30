# airqualityapp/views.py
from django.shortcuts import render
from .pyspark_service import perform_air_quality_analysis
import pandas as pd
from django.http import JsonResponse
from celery.result import AsyncResult
from django.views.decorators.csrf import csrf_exempt
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import os
from django.conf import settings
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.shortcuts import render
from django.http import JsonResponse
from .utils import *
from django.shortcuts import render


'''
premiere partie :
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Itération sur les fichiers dans le répertoire spécifié et affichage de leur chemin
for dirname, _, filenames in os.walk('../input/solar-power-generation-data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Chargement des données
generation_data = pd.read_csv('C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/Plant_1_Generation_Data.csv')
weather_data = pd.read_csv('C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/Plant_1_Weather_Sensor_Data.csv')

# Conversion des colonnes DATE_TIME en datetime
generation_data['DATE_TIME'] = pd.to_datetime(generation_data["DATE_TIME"])
weather_data['DATE_TIME'] = pd.to_datetime(weather_data["DATE_TIME"])

# Fusion des DataFrames
df = pd.merge(generation_data.drop(columns=['PLANT_ID']), weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# Encodage de la colonne SOURCE_KEY
encoder = LabelEncoder()
df['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df['SOURCE_KEY'])

# Sauvegarde du DataFrame fusionné dans un fichier CSV
df.to_csv('C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/Merged_Solar_Data.csv', index=False)


new = pd.read_csv('C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/Merged_Solar_Data.csv')


def prediction_view(request):
    return render(request, 'prediction.html')



# focntion permettant de predire letat de l'air a laide la fonction de calcule d'index aqi 
def predict_air_quality_view(request):
    # Load your models
    model_paths = {
        'pls_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/pls_model.pkl',
        'dtr_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/dtr_model.pkl',
        'lr_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/lr_model.pkl',
        'knn_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/knn_model.pkl',
        'rfr_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/rfr_model.pkl',
        'rnn_model': 'C:/Users/kaout/OneDrive/Documents/Python Scripts/plateform/model/rnn_model.h5'
    }
    import tensorflow as tf
    import joblib

    pls_model = joblib.load(model_paths['pls_model'])
    dtr_model = joblib.load(model_paths['dtr_model'])
    lr_model = joblib.load(model_paths['lr_model'])
    knn_model = joblib.load(model_paths['knn_model'])
    rfr_model = joblib.load(model_paths['rfr_model'])
    rnn_model = tf.keras.models.load_model(model_paths['rnn_model'])

    if request.method == 'POST':
        # Retrieve form data
        Daily_Yield = float(request.POST.get('Daily_Yield'))
        Total_yield = float(request.POST.get('Total_yield'))
        ambient_temp = float(request.POST.get('ambient_temp'))
        module_temp = float(request.POST.get('module_temp'))
        irradiation = float(request.POST.get('irradiation'))

        # Create a DataFrame with the data
        data = {
            'DAILY_YIELD': [Daily_Yield],
            'TOTAL_YIELD': [Total_yield],
            'AMBIENT_TEMPERATURE': [ambient_temp],
            'MODULE_TEMPERATURE': [module_temp],
            'IRRADIATION': [irradiation]
        }
        df = pd.DataFrame(data)

        # Convertit le DataFrame en tensor
        data = tf.convert_to_tensor(df.values, dtype=tf.float32)

        # Ajoute une dimension
        data = tf.expand_dims(data, axis=1)  # La forme devient (nombre_d'échantillons, 1, nombre_de_features)

        # Make predictions
        prediction_results = {
            'prediction_result_pls': pls_model.predict(df)[0],
            'r2_score_pls': '97.7 %',
            'prediction_result_dtr': dtr_model.predict(df)[0],
            'r2_score_dtr': '98.5 %',
            'prediction_result_lr': lr_model.predict(df)[0],
            'r2_score_lr': '98 %',
            'prediction_result_knn': knn_model.predict(df)[0],
            'r2_score_knn': '86.9 %',
            'prediction_result_rfr': rfr_model.predict(df)[0],
            'r2_score_rfr': '99 %',
            'prediction_result_rnn': rnn_model.predict(data)[0],
            'r2_score_rnn': '98.4 %'
        }

        # Render the results to the template
        return render(request, 'prediction_interface2.html', prediction_results)

    return render(request, 'prediction_form.html')

# pour la generation du graphe(le chart)

def training_and_test(request):
    # Chemin vers ,otre fichier CSV

    # Chargement  des données depuis le CSV
    

    # Comptage des occurrences de chaque état
    state_counts = df['AQI_Bucket'].value_counts()

    # Calcule des pourcentages
    total_states = len(df['AQI_Bucket'])
    state_percentages = (state_counts / total_states) * 100

    # Création d un graphe circulaire avec les données spécifiées
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    sizes = state_percentages.tolist()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'yellow', 'orange', 'red', 'purple', 'maroon'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    #  une légende
    legend = ax.legend(title='Air Quality States', loc='center left', bbox_to_anchor=(1, 0.5))
    for i, text in enumerate(legend.get_texts()):
        text.set_text(f'{labels[i]} ')

    # Sauvegardons le graphe dans un objet BytesIO
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    image_stream.seek(0)

    # on va Convertir l'image en base64 pour l'inclure dans le template
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    image_url = f'data:image/png;base64,{image_base64}'

    # on passe  l'URL de l'image et les pourcentages à la page
    context = {'image_url': image_url, 'state_percentages': state_percentages}
    return render(request, 'training_and_test.html', context)



# ctte methode concernne le traitement et analyse  des donnés generées 

def air_quality_analysis(request):
    # Spécifiez le chemin vers votre fichier CSV
    csv_path = "C:/Users/kaout/OneDrive/Documents/Python Scripts/city_day.csv"

    # Appele au  service PySpark
    result_data = perform_air_quality_analysis(csv_path)

    
    return render(request, 'analysis_result.html', {'result_data': result_data})




# ctte methode permet de lire le fichier teelcharger
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        # chemin complet pour le fichier dans le répertoire 'media/uploads'
        csv_file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', csv_file.name)

        # Enregistrement du  fich  dans le répertoire 'media/uploads'
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Chargement  du fich CSV dans un DataFrame pandas
        df = pd.read_csv(csv_file_path)

        # Renvoyer la prévisualisation du DataFrame
        return render(request, 'preview.html', {'table_html': df.to_html()})

    return render(request, 'upload.html')

#pour l'affichage des donnés sous forma d'une table 

def preview_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # Récupérer le fichier CSV depuis la requête POST
        csv_file = request.FILES['csv_file']

        # Construire le chemin complet pour le fichier
        csv_file_path = os.path.join('media', 'uploads', csv_file.name)
        
        # Enregistrez le fichier dans le répertoire 'media/uploads/'
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Chargement du fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_file_path)

        # on va convertir le DataFrame en HTML pour l'affichage dans le modèle
        table_html = df.to_html()

        # Envoyer le HTML à la page de prévisualisation
        return render(request, 'preview.html', {'table_html': table_html})

    # Si la méthode n'est pas POST ou si aucun fichier n'est fourni, revenir à la page d'accueil
    return render(request, 'home.html')








def check_task_status(request, task_id):
    result = AsyncResult(task_id)

    if result.ready():
        # Si La tâche est terminée, renvoyer le résultat
        table_html = result.result
        return JsonResponse({'status': 'SUCCESS', 'table_html': table_html})
    elif result.state == 'PENDING':
        #Si La tâche est en attente
        return JsonResponse({'status': 'PENDING'})
    else:
        # Si La tâche a échoué ou a une autre statut, gérer en conséquence
        return JsonResponse({'status': 'FAILURE'})


@csrf_exempt
def preview_csv_async(request):
    # Cette vue est utilisée pour initier la connexion WebSocket
    return render(request, 'preview.html')


def send_table(event):
    # Cette fonction est appelée lorsqu'un message est envoyé au groupe
    table_html = event['table_html']

    # Envoyer la table aux clients connectés via WebSocket
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "preview_group", {"type": "send.table", "table_html": table_html}
    )


def home(request):
    return render(request, 'home.html')
def upload(request):
    return render(request, 'upload.html')