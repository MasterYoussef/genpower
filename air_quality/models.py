# airqualityapp/models.py
from django.db import models

class AirQualityData(models.Model):
    timestamp = models.DateTimeField()
    pollutant1 = models.FloatField()
    pollutant2 = models.FloatField()
    # Ajoutez d'autres champs en fonction de vos besoins
class YourModel(models.Model):
    csv_file = models.FileField(upload_to='uploads/')