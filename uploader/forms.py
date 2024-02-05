from django import forms
from .models import Ancient_Image

class Image_Form(forms.ModelForm):
    class Meta:
        model = Ancient_Image
        fields = [
        'picture'
        ]