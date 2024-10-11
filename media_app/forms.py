# media_app/forms.py
from django import forms
from .models import VideoBase, ImageBase


class VideoForm(forms.ModelForm):
    class Meta:
        model = VideoBase
        fields = ['video']


class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageBase
        fields = ['image']


class MultipleImageForm(forms.Form):
    images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
