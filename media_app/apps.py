from django.apps import AppConfig
import os

class MediaAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'  # 适配 Django 3.2+，可选
    name = 'media_app'

    # 这里指定绝对路径，确保唯一性
    path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
