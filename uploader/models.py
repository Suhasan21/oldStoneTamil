from django.db import models
class Ancient_Image(models.Model):
    letter = models.CharField(max_length=200)
    picture = models.FileField()
    #verify = models.BooleanField()
    
    def __str__(self):
        return self.letter

class Verify(models.Model):
    image = models.OneToOneField(
        Ancient_Image,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    verify = models.BooleanField(default=False)

    def __str__(self):
        return self.verify