# media_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .forms import VideoForm, ImageForm

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return JsonResponse({'message': 'Video uploaded successfully'}, status=200)
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return JsonResponse({'message': 'Image uploaded successfully'}, status=200)
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})
