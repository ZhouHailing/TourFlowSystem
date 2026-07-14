from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video/', views.video_stream, name='video'),
    path('get_count/', views.get_count, name='get_count'),
    path('reset/', views.reset_session, name='reset'),
    path('set_threshold/', views.set_threshold, name='set_threshold'),
    path('face/', views.face_list, name='face'),
    path('warn/', views.warn_list, name='warn'),
    path('get_chart_data/', views.get_chart_data, name='get_chart_data'),
    path('get_weather/', views.get_weather, name='get_weather')
]
