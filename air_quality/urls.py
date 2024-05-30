# airqualityapp/urls.py
from django.urls import path
from .views import home,upload,air_quality_analysis, check_task_status,upload_csv
from . import views
from .views import predict_air_quality_view,prediction_view

urlpatterns = [
    path('', home, name='home'),
    path('upload/', upload, name='upload'),
    path('upload/', upload_csv, name='upload_csv'),
    path('air_quality_analysis/', air_quality_analysis, name='air_quality_analysis'),
    path('preview/', views.preview_csv, name='preview_csv'),
    path('check_task_status/<str:task_id>/', check_task_status, name='check_task_status'),

    path('training_and_test/', views.training_and_test, name='training_and_test'),
     path('predict_air_quality/', predict_air_quality_view, name='predict_air_quality_view'),
      path('prediction/', prediction_view, name='prediction_view'),



]
