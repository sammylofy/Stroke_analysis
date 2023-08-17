from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class UserDet(models.Model):
    phone = models.CharField(primary_key=True, max_length=15, null=False, unique=True)
    name = models.CharField(unique=False, null=False, max_length=100)

    class Meta:
        db_table = "userdet"


class DocDet(models.Model):
    userid = models.ForeignKey(User, on_delete=models.CASCADE)
    phone = models.CharField(unique=False, null=False, max_length=15)

    class Meta:
        db_table = "docdet"


class Data(models.Model):
    did = models.CharField(primary_key=True, max_length=100, unique=True)
    sex = models.IntegerField(null=False)
    age = models.IntegerField(null=False)
    history = models.IntegerField(null=False)
    hypertension = models.IntegerField(null=False)
    inactivity = models.IntegerField(null=False)
    cardiovascular = models.IntegerField(null=False)
    hyperlidermia = models.IntegerField(null=False)
    alcohol = models.IntegerField(null=False)
    tia = models.IntegerField(null=False)
    msyndrome = models.IntegerField(null=False)
    atherosclerosis = models.IntegerField(null=False)
    af = models.IntegerField(null=False)
    lvh = models.IntegerField(null=False)
    diabetes = models.IntegerField(null=False)
    smoking = models.IntegerField(null=False)
    stroke = models.IntegerField(null=False)
    phone = models.ForeignKey(UserDet, on_delete=models.CASCADE)
    advice = models.TextField()

    class Meta:
        db_table = "data"

class Researchers(models.Model):
    email = models.CharField(unique=False, null=False, max_length=100)
    name = models.CharField(unique=False, null=False, max_length=100)
    date = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "researchers"