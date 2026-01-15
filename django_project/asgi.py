"""
ASGI config for django_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import media_app.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yourproject.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    # HTTP 协议使用 Django 默认的 ASGI 应用
    "http": django_asgi_app,

    # WebSocket 协议
    "websocket": AuthMiddlewareStack(
        URLRouter(
            media_app.routing.websocket_urlpatterns
        )
    ),
})
