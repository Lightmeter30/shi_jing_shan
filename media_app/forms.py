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


# class MultipleImageForm(forms.Form):
#   images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
