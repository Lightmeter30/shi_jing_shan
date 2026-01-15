from django.urls import re_path
from .consumers import TaskProgressConsumer

websocket_urlpatterns = [
    re_path(r"^ws/task/(?P<task_id>[^/]+)/$", TaskProgressConsumer.as_asgi()),
]