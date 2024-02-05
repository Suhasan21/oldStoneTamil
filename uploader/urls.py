from django.urls import path
from . import views
from Recognition import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.create_image_uploader, name = 'create'),
    path('verify_image/<int:id>', views.verify_image, name='verify_image'),
    path('create_image_uploader/',views.create_image_uploader) ,
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
