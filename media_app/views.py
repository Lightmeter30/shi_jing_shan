# media_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .models import Image, Video
from .forms import VideoForm, ImageForm
from utils.upload import new_name
import os


def upload_video(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'videos/')
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.video.name = new_name(custom_location,
                                           instance.video.name)
            # Assign the custom location to the image instance
            instance.save()
            return JsonResponse({'message': 'Video uploaded successfully'},
                                status=200)
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})


def upload_image(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'images/')
        # print(custom_location)
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save(commit=False)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name)
            # Assign the custom location to the image instance
            image_instance.save()
            return JsonResponse({'message': 'Image uploaded successfully'},
                                status=200)
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})


def upload_multiple_images(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        custom_location = request.GET.get('custom_location', 'images/')
        for image_file in request.FILES.getlist('images'):
            image_instance = Image(image=image_file)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name)
            image_instance.save()
        return JsonResponse(
            {'message': 'Images uploaded successfully to custom location'},
            status=200)
    return JsonResponse({'error': 'POST request and images required'},
                        status=400)


def upload_multiple_videos(request):
    if request.method == 'POST' and request.FILES.getlist('videos'):
        custom_location = request.GET.get('custom_location', 'videos/')
        for video_file in request.FILES.getlist('videos'):
            instance = Video(video=video_file)
            instance.video.name = new_name(custom_location,
                                           instance.video.name)
            instance.save()
        return JsonResponse({'message': 'Videos uploaded successfully'},
                            status=200)
    return JsonResponse({'error': 'POST request and videos required'},
                        status=400)


def request_colmap(request):
    if request.method == 'GET':
        request_location = request.GET.get('request_location', 'images/')
        save_location = request.GET.get('save_location', 'colmaps/')
        colmap_params = request.GET.get('colmap_params', '')
        folder = Image.objects.filter(image__contains=request_location)
        if folder:
            os.system('colmap --input_path ' +
                      os.path.join(settings.MEDIA_ROOT, request_location) +
                      ' --output_path ' +
                      os.path.join(settings.MEDIA_ROOT, save_location) + ' ' +
                      colmap_params)
            return JsonResponse({'messag e': 'Folder found'}, status=200)
        else:
            return JsonResponse(
                {'error': 'No images found in the specified folder'},
                status=404)

    return JsonResponse({'error': 'Get request required'}, status=400)
