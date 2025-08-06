# media_app/models.py
from django.db import models
import os
import uuid
from django_project import settings


class Video(models.Model):
    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Dataset(models.Model):
    name = models.CharField(max_length=255, unique=True)
    updated_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=255) # 相对路径
    info = models.JSONField(default=dict)  # 存储额外信息，如描述、标签等
    config = models.JSONField(default=dict)  # 存储相关配置
    old_config = models.JSONField(default=dict)  # 存储旧的配置参数

class DatasetFile(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='files', on_delete=models.CASCADE)
    name = models.CharField(max_length=255, blank=True)
    file_path = models.CharField(max_length=255) # 相对路径
    file_type = models.CharField(max_length=50)  # e.g., 'object', 'ply'
    uploaded_at = models.DateTimeField(auto_now_add=True)