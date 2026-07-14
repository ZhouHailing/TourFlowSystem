from django.contrib import admin
from .models import FaceCapture, FlowCount, FlowWarn


@admin.register(FaceCapture)
class FaceCaptureAdmin(admin.ModelAdmin):
    list_display = ('scenic_point', 'capture_time')
    list_filter = ('scenic_point',)


@admin.register(FlowCount)
class FlowCountAdmin(admin.ModelAdmin):
    list_display = ('stat_time', 'in_num', 'out_num', 'current_people')


@admin.register(FlowWarn)
class FlowWarnAdmin(admin.ModelAdmin):
    list_display = ('warn_time', 'warn_people', 'warn_threshold', 'warn_content')
