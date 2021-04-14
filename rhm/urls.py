from django.urls import path
from . import views

app_name = 'rhm'
urlpatterns = [
    path('', views.home, name='home'),
    path('detail/', views.get_details, name='details'),
]