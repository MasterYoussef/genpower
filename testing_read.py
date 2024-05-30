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


# Affichage des premières lignes du DataFrame fusionné pour vérifier
print(new.head())
