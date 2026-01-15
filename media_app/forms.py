# media_app/forms.py
from django import forms
from .models import Video, Image, Dataset, DatasetFile


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file_path', 'info', 'config', 'old_config']


class DatasetFileForm(forms.ModelForm):
    class Meta:
        model = DatasetFile
        fields = ['dataset', 'file_path', 'file_type']


# class MultipleImageForm(forms.Form):
#   images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
