# ai_app/urls.py
from django.urls import path
from . import views  # Import your view functions here

urlpatterns = [
    path('chat/', views.chat_to_ollama, name='ai_endpoint'),  # Define your URL patterns here
]