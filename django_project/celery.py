import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project.settings')

app = Celery('django_project')

# 从 Django 的 settings.py 里加载 CELERY 配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现所有 app 里的 tasks.py
app.autodiscover_tasks()
