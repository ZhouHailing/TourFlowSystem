from django import forms


class ScenicPointForm(forms.Form):
    scenic_point = forms.CharField(max_length=50, label='抓拍点位')
