from predictor.models import Data, UserDet
from django import template

register = template.Library()

@register.simple_tag
def history(condition):
    try:
        total= Data.objects.filter(history=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def hypertension(condition):
    try:
        total= Data.objects.filter(hypertension=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def stroke(condition):
    try:
        total= Data.objects.filter(stroke=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def af(condition):
    try:
        total= Data.objects.filter(af=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def diabetes(condition):
    try:
        total= Data.objects.filter(diabetes=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def smoking(condition):
    try:
        total= Data.objects.filter(smoking=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def sex(condition):
    try:
        total= Data.objects.filter(sex=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def hyperlidermia(condition):
    try:
        total= Data.objects.filter(hyperlidermia=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def tia(condition):
    try:
        total= Data.objects.filter(tia=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def msyndrome(condition):
    try:
        total= Data.objects.filter(msyndrome=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def atherosclerosis(condition):
    try:
        total= Data.objects.filter(atherosclerosis=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def sex(condition):
    try:
        total= Data.objects.filter(sex=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def hyperlidermia(condition):
    try:
        total= Data.objects.filter(hyperlidermia=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def alcohol(condition):
    try:
        total= Data.objects.filter(alcohol=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def inactivity(condition):
    try:
        total= Data.objects.filter(inactivity=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def cardiovascular(condition):
    try:
        total= Data.objects.filter(cardiovascular=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0
@register.simple_tag
def lvh(condition):
    try:
        total= Data.objects.filter(lvh=condition).count()
        return total
        print(total)
    except Data.DoesNotExist:
        return 0

@register.simple_tag
def getname(phone):
    try:
        uname = UserDet.objects.values('name').get(phone=phone)
        name = uname.get('name')
        return name
    except UserDet.DoesNotExist:
        return None