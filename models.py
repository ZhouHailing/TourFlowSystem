from django.db import models
import datetime


class FaceCapture(models.Model):
    face_img = models.ImageField(upload_to='face_img/', verbose_name="抓拍人脸图片")
    capture_time = models.DateTimeField(default=datetime.datetime.now, verbose_name="抓拍时间")
    scenic_point = models.CharField(max_length=50, default="景区正门", verbose_name="抓拍点位")

    class Meta:
        verbose_name = "人脸抓拍记录"
        verbose_name_plural = verbose_name
        ordering = ['-capture_time']

    def __str__(self):
        return f"{self.scenic_point}-{self.capture_time.strftime('%Y-%m-%d %H:%M:%S')}"


class FlowCount(models.Model):
    in_num = models.IntegerField(default=0, verbose_name="进场人数")
    out_num = models.IntegerField(default=0, verbose_name="离场人数")
    stat_time = models.DateTimeField(default=datetime.datetime.now, verbose_name="统计时间")
    current_people = models.IntegerField(default=0, verbose_name="当前在园客流")

    class Meta:
        verbose_name = "客流分时统计"
        verbose_name_plural = verbose_name
        ordering = ['-stat_time']


class FlowWarn(models.Model):
    warn_people = models.IntegerField(verbose_name="超限客流数量")
    warn_threshold = models.IntegerField(default=50, verbose_name="预警阈值")
    warn_time = models.DateTimeField(default=datetime.datetime.now, verbose_name="预警触发时间")
    warn_content = models.CharField(max_length=200, verbose_name="预警详情")

    class Meta:
        verbose_name = "客流超限预警"
        verbose_name_plural = verbose_name
        ordering = ['-warn_time']
