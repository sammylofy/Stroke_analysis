from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login_user/', views.login_user, name='login_user'),
    path('index/', views.index, name='index'),
    path('clients/', views.clients, name='clients'),
    path('message/', views.message, name='message'),
    path('report/', views.report, name='report'),
    path('viewallreports/', views.viewallreports, name='viewallreports'),
    path('dataset/', views.dataset, name='dataset'),
    path('deluser/<id>', views.deluser, name='deluser'),
    path('printreport/<pid>', views.printreport, name='printreport'),
    path('printureport/<id>', views.printureport, name='printureport'),
    path('detailreport/<did>', views.detailreport, name='detailreport'),
    path('users/', views.users, name='users'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard1/', views.dashboard1, name='dashboard1'),
    path('requestdataset/', views.requestdataset, name='requestdataset'),
    path('register/', views.register, name='register'),
    path('logout_user/', views.logout_user, name='logout_user'),
    path('about/', views.about, name='about'),
    path('age/', views.age, name='age'),
    path('sex/', views.sex, name='sex'),
    path('question1/', views.question1, name='question1'),
    path('question2/', views.question2, name='question2'),
    path('question3/', views.question3, name='question3'),
    path('question4/', views.question4, name='question4'),
    path('question5/', views.question5, name='question5'),
    path('question6/', views.question6, name='question6'),
    path('question7/', views.question7, name='question7'),
    path('question8/', views.question8, name='question8'),
    path('question9/', views.question9, name='question9'),
    path('question10/', views.question10, name='question10'),
    path('question11/', views.question11, name='question11'),
    path('question12/', views.question12, name='question12'),
    path('question13/', views.question13, name='question13'),
    path('predict/', views.predict, name='predict'),
    path('results/<conseling>/<pred>/<phone>/<prob>/<result>/<id>', views.results, name='results'),

]
