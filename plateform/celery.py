# celery.py

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Réglez le module Django pour que Celery découvre l'application Django.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plateform.settings')

# Créez une instance de l'application Celery et configurez-la en utilisant le fichier settings de Django.
app = Celery('plateform')

# Chargez les configurations de manière asynchrone pour que les tâches puissent utiliser Django.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Découvrez et chargez automatiquement les tâches dans tous les modules "tasks.py" de l'application Django.
app.autodiscover_tasks()
