# media_app/models.py
from django.db import models
import os
import uuid


class Video(models.Model):
    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
