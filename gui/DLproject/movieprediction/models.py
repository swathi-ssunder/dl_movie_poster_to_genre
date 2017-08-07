from django.db import models

from django.forms import ModelForm

class Upload(models.Model):
    pic = models.ImageField("Image", upload_to="")
    upload_date=models.DateTimeField(auto_now_add =True)

# FileUpload form class.
class UploadForm(ModelForm):
    class Meta:
        model = Upload
        fields = "__all__"
