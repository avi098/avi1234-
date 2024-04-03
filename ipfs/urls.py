from django.urls import path
from . import views

urlpatterns = [
    path('submit_text/', views.submit_text, name='submit_text'),
    path('view_text/', views.view_text, name='view_text'),
]
