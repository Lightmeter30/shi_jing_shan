# media_app/forms.py
from django import forms
from .models import Video, Image

class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']
