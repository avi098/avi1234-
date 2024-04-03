from django.db import models


class datas(models.Model):
    data1 = models.CharField(max_length=600)
    data2 = models.CharField(max_length=600)
    data3 = models.CharField(max_length=600)
    data4 = models.CharField(max_length=600)
    data5 = models.CharField(max_length=600)

class UserProfile(models.Model):
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=50)
    

    def __str__(self):
        return str(self.pk)
