from django.shortcuts import render
from django.http import JsonResponse


# Create your views here.
def query_to_ollama(request):
    # Implement your AI-related logic here
    return JsonResponse({'message': 'AI endpoint response'})