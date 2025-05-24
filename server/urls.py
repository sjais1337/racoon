from django.contrib import admin
from django.urls import path, include
from accounts import views as accounts_views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('', accounts_views.home, name='home'), 
    path('dashboard/', accounts_views.dashboard, name='dashboard'),
    path('groups/', include('fl_platform.urls')), 
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)