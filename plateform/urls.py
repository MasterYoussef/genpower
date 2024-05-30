# urls.py dans le répertoire principal du projet
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('air_quality.urls')),
]
