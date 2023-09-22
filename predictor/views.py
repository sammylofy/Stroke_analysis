import csv
import pickle
import random
import requests.exceptions

import os
import openai
from django.contrib import auth, messages
from django.contrib.auth.models import User
from django.http import request, HttpResponse, HttpResponseServerError
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.urls import reverse
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .models import UserDet, Data, DocDet, Researchers
from django.utils import timezone
from requests import session
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = "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"


# Create your views here.

def login_user(request):
    title = 'Login'
    context = {"title": title}
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                auth.login(request, user)
                if user.is_superuser:
                    return redirect('dashboard1')
                else:
                    return redirect('dashboard')
            else:
                messages.info(request, 'Your account is inactive and cannot login')
                return redirect('login_user')
        else:
            messages.info(request, 'Invalid Username or Password')
            return redirect('login_user')
    else:
        return render(request, 'login.html', context)


def logout_user(request):
    auth.logout(request)
    return redirect('index')


def home(request):
    return redirect('index')


def index(request):
    title = "Home"
    return render(request, 'index.html', context={'title': title})


##@login_required(login_url='/login_user')
def dashboard(request):
    if request.user.is_authenticated:
        title = "Dashboard"
        return render(request, 'dashboard.html', context={'title': title})
    else:
        return redirect('login_user')


##@login_required(login_url='/login_user')
def dashboard1(request):
    if request.user.is_authenticated:
        title = "Dashboard"
        users = DocDet.objects.all()
        return render(request, 'dashboard1.html', context={'title': title, "users": users})
    else:
        return redirect('login_user')


# @login_required(login_url='/login_user')
def clients(request):
    if request.user.is_authenticated:
        title = "Clients"
        try:
            clients = UserDet.objects.all()
        except UserDet.DoesNotExist:
            clients = None
        return render(request, 'clients.html', context={'title': title, "clients": clients})
    else:
        return redirect('login_user')


# @login_required(login_url='/login_user')
def report(request):
    if request.user.is_authenticated:
        title = "Reports"
        try:
            reports = Data.objects.all()
        except Data.DoesNotExist:
            reports = None
        return render(request, 'reports.html', context={'title': title, "reports": reports})
    else:
        return redirect('login_user')


# @login_required(login_url='/login_user')
def detailreport(request, did):
    if request.user.is_authenticated:
        title = "Detail Report"
        gender = Data.objects.values('sex').get(did=did)
        sex = gender.get('sex')
        cgender = ""
        if sex == 1:
            cgender = "Male"
        elif sex == 0:
            cgender = "Female"
        cage = Data.objects.values('age').get(did=did)
        age = cage.get('age')
        chistory = Data.objects.values('history').get(did=did)
        history = chistory.get('history')
        chypertension = Data.objects.values('hypertension').get(did=did)
        hypertension = chypertension.get('hypertension')
        cinactivity = Data.objects.values('inactivity').get(did=did)
        inactivity = cinactivity.get('inactivity')
        ccardiovascular = Data.objects.values('cardiovascular').get(did=did)
        cardiovascular = ccardiovascular.get('cardiovascular')
        chyperlidermia = Data.objects.values('hyperlidermia').get(did=did)
        hyperlidermia = chyperlidermia.get('hyperlidermia')
        calcohol = Data.objects.values('alcohol').get(did=did)
        alcohol = calcohol.get('alcohol')
        ctia = Data.objects.values('tia').get(did=did)
        tia = ctia.get('tia')
        cmsyndrome = Data.objects.values('msyndrome').get(did=did)
        msyndrome = cmsyndrome.get('msyndrome')
        catherosclerosis = Data.objects.values('atherosclerosis').get(did=did)
        atherosclerosis = catherosclerosis.get('atherosclerosis')
        caf = Data.objects.values('af').get(did=did)
        af = caf.get('af')
        clvh = Data.objects.values('lvh').get(did=did)
        lvh = clvh.get('lvh')
        cdiabetes = Data.objects.values('diabetes').get(did=did)
        diabetes = cdiabetes.get('diabetes')
        csmoking = Data.objects.values('smoking').get(did=did)
        smoking = csmoking.get('smoking')
        cstroke = Data.objects.values('stroke').get(did=did)
        stroke = cstroke.get('stroke')
        cadvice = Data.objects.values('advice').get(did=did)
        advice = cadvice.get('advice')
        cphone = Data.objects.values('phone_id').get(did=did)
        phone = cphone.get('phone_id')
        model = pickle.load(open('model.pkl', 'rb'))
        probability = model.predict_proba([
            [history, hypertension, inactivity, cardiovascular, hyperlidermia, alcohol, tia, msyndrome, atherosclerosis,
             sex, age, af,
             lvh, diabetes, smoking]])
        prob = probability[0][1]
        prob = float(prob * 100)
        prob =round(prob,2)

        return render(request, 'detailreport.html',
                      context={'title': title, "sex": sex, "age": age, "history": history, "hypertension": hypertension,
                               "inactivity": inactivity, "cardiovascular": cardiovascular,
                               "hyperlidermia": hyperlidermia, "alcohol": alcohol, "tia": tia, "msyndrome": msyndrome,
                               "atherosclerosis": atherosclerosis, "af": af, "lvh": lvh,
                               "diabetes": diabetes, "smoking": smoking, "stroke": stroke, "percent": prob,
                               "phone": phone,
                               "advice": advice, "cgender": cgender, "did": did})
    else:
        return redirect('login_user')


# @login_required(login_url='/login_user')
def printreport(request, did):
    if request.user.is_authenticated:
        title = "Print Report"
        gender = Data.objects.values('sex').get(did=did)
        sex = gender.get('sex')
        cgender = ""
        if sex == 1:
            cgender = "Male"
        elif sex == 0:
            cgender = "Female"
        cage = Data.objects.values('age').get(did=did)
        age = cage.get('age')
        chistory = Data.objects.values('history').get(did=did)
        history = chistory.get('history')
        chypertension = Data.objects.values('hypertension').get(did=did)
        hypertension = chypertension.get('hypertension')
        cinactivity = Data.objects.values('inactivity').get(did=did)
        inactivity = cinactivity.get('inactivity')
        ccardiovascular = Data.objects.values('cardiovascular').get(did=did)
        cardiovascular = ccardiovascular.get('cardiovascular')
        chyperlidermia = Data.objects.values('hyperlidermia').get(did=did)
        hyperlidermia = chyperlidermia.get('hyperlidermia')
        calcohol = Data.objects.values('alcohol').get(did=did)
        alcohol = calcohol.get('alcohol')
        ctia = Data.objects.values('tia').get(did=did)
        tia = ctia.get('tia')
        cmsyndrome = Data.objects.values('msyndrome').get(did=did)
        msyndrome = cmsyndrome.get('msyndrome')
        catherosclerosis = Data.objects.values('atherosclerosis').get(did=did)
        atherosclerosis = catherosclerosis.get('atherosclerosis')
        caf = Data.objects.values('af').get(did=did)
        af = caf.get('af')
        clvh = Data.objects.values('lvh').get(did=did)
        lvh = clvh.get('lvh')
        cdiabetes = Data.objects.values('diabetes').get(did=did)
        diabetes = cdiabetes.get('diabetes')
        csmoking = Data.objects.values('smoking').get(did=did)
        smoking = csmoking.get('smoking')
        cstroke = Data.objects.values('stroke').get(did=did)
        stroke = cstroke.get('stroke')
        cadvice = Data.objects.values('advice').get(did=did)
        advice = cadvice.get('advice')
        cphone = Data.objects.values('phone_id').get(did=did)
        phone = cphone.get('phone_id')
        model = pickle.load(open('model.pkl', 'rb'))
        probability = model.predict_proba([
            [history, hypertension, inactivity, cardiovascular, hyperlidermia, alcohol, tia, msyndrome, atherosclerosis,
             sex, age, af,
             lvh, diabetes, smoking]])
        prob = probability[0][1]
        prob = float(prob * 100)
        prob = round(prob, 2)

        return render(request, 'printreport.html',
                      context={'title': title, "sex": sex, "age": age, "history": history, "hypertension": hypertension,
                               "inactivity": inactivity, "cardiovascular": cardiovascular,
                               "hyperlidermia": hyperlidermia, "alcohol": alcohol, "tia": tia, "msyndrome": msyndrome,
                               "atherosclerosis": atherosclerosis, "af": af, "lvh": lvh,
                               "diabetes": diabetes, "smoking": smoking, "stroke": stroke, "percent": prob,
                               "phone": phone,
                               "advice": advice, "cgender": cgender, "did": did})
    else:
        return redirect('login_user')


def dataset(request):
    title = "Clients"
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="strokedataset.csv"'
    writer = csv.writer(response)
    writer.writerow(
        ['ID', 'Sex', 'Age', 'Family History', 'Hypertension', "Physical Inactivity", "Cardiovascular Disease",
         "Hyperlidermia",
         "Alcohol Comsuption", "History of TIA", "Metabolic Syndrome", "Atherosclerosis", "Atrial Fibrillation",
         "Left Ventricular Hypertrophy", "Diabetes",
         "Smoking", "Stroke"])
    try:
        dataset = Data.objects.all()

        for data in dataset:
            writer.writerow(
                [data.did, data.sex, data.age, data.history, data.hypertension, data.inactivity, data.cardiovascular,
                 data.hyperlidermia, data.alcohol, data.tia, data.msyndrome, data.atherosclerosis, data.af, data.lvh,
                 data.diabetes, data.smoking, data.stroke])
    except Data.DoesNotExist:
        writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return response


# @login_required(login_url='/login_user')
def users(request):
    if request.user.is_authenticated:
        title = 'New User'
        # users = User.objects.filter(is_superuser=False)
        users = DocDet.objects.all()
        context = {'title': title, "users": users}
        if request.method == "POST":
            username = request.POST.get('username')
            password = request.POST.get('password')
            cpassword = request.POST.get('cpassword')
            firstname = request.POST.get('firstname')
            lastname = request.POST.get('lastname')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            if password == cpassword:
                if User.objects.filter(username=username).exists():
                    messages.info(request, 'Username already exist')
                    return redirect('users')
                else:
                    password = make_password(password)
                    u = User.objects.create(first_name=firstname, password=password, is_superuser=False,
                                            username=username,
                                            last_name=lastname, email=email, is_staff=True, is_active=True,
                                            date_joined=timezone.now())
                    u.save()
                    user = User.objects.get(username=username)
                    userid = user.id
                    ph = DocDet.objects.create(userid_id=userid, phone=phone)
                    ph.save()
                    messages.info(request, 'User added successfully')
                    return redirect(reverse('users'))
            else:
                messages.info(request, 'Password mismatch error')
                return redirect('users')
        else:
            return render(request, 'users.html', context)
    else:
        return redirect('login_user')


# @login_required(login_url='/login_user')
def deluser(request, id):
    try:
        user = User.objects.filter(id=id).delete()
        return redirect('users')
    except User.DoesNotExist:
        messages.info(request, 'User does not exist')
        return redirect('users')


def register(request):
    title = 'User Details'
    context = {'title': title}
    if request.method == "POST":
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        if UserDet.objects.filter(phone=phone).exists():
            request.session['phone'] = phone
            return redirect('age')
        else:
            udet = UserDet.objects.create(phone=phone, name=name)
            request.session['phone'] = phone
            return redirect('age')
    return render(request, 'register.html', context)


def about():
    title = "About"
    return render(request, 'about.html', context={'title': title})


def message(request):
    title = "Message"
    return render(request, 'message.html', context={'title': title})


def age(request):
    title = "Age"
    if request.method == 'POST':
        # age = request.form['1']
        request.session["age"] = request.POST.get("age")
        # request.session = qst1
        return redirect('sex')
    return render(request, 'age.html', context={'title': title})


def sex(request):
    title = "Gender"
    if request.method == 'POST':
        request.session["sex"] = request.POST.get("sex")
        # request.session = qst1
        return redirect('question1')
    return render(request, 'gender.html', context={'title': title})


def question1(request):
    if request.method == 'POST':
        # qst1 = request.form['1']
        request.session["qst1"] = request.POST.get("1")
        # request.session = qst1
        return redirect('question2')
    return render(request, 'question1.html')


def question2(request):
    if request.method == 'POST':
        request.session["qst2"] = request.POST.get("2")
        # request.session = qst1
        return redirect('question3')
    return render(request, 'question2.html')


def question3(request):
    if request.method == 'POST':
        request.session["qst3"] = request.POST.get("3")
        return redirect('question4')
    return render(request, 'question3.html')


def question4(request):
    if request.method == 'POST':
        request.session["qst4"] = request.POST.get("4")
        return redirect('question5')
    return render(request, 'question4.html')


def question5(request):
    if request.method == 'POST':
        request.session["qst5"] = request.POST.get("5")
        return redirect('question6')
    return render(request, 'question5.html')


def question6(request):
    if request.method == 'POST':
        request.session["qst6"] = request.POST.get("6")
        return redirect('question7')
    return render(request, 'question6.html')


def question7(request):
    if request.method == 'POST':
        request.session["qst7"] = request.POST.get("7")
        return redirect('question8')
    return render(request, 'question7.html')


def question8(request):
    if request.method == 'POST':
        request.session["qst8"] = request.POST.get("8")
        return redirect('question9')
    return render(request, 'question8.html')


def question9(request):
    if request.method == 'POST':
        request.session["qst9"] = request.POST.get("9")
        return redirect('question10')
    return render(request, 'question9.html')


def question10(request):
    if request.method == 'POST':
        request.session["qst10"] = request.POST.get("10")
        return redirect('question11')
    return render(request, 'question10.html')


def question11(request):
    if request.method == 'POST':
        request.session["qst11"] = request.POST.get("11")
        return redirect('question12')
    return render(request, 'question11.html')


def question12(request):
    if request.method == 'POST':
        request.session["qst12"] = request.POST.get("12")
        return redirect('question13')
    return render(request, 'question12.html')


def question13(request):
    if request.method == 'POST':
        request.session["qst13"] = request.POST.get("13")
        return redirect('predict')
    return render(request, 'question13.html')


def predict(request):
    # collect prediction data from client
    random_number = random.randint(1000000000, 9999999999)
    id = str(random_number)
    history = int(request.session["qst1"])
    hypertension = int(request.session["qst2"])
    inactivity = int(request.session["qst3"])
    cardiovascular = int(request.session["qst4"])
    hyperlidermia = int(request.session["qst5"])
    alcohol = int(request.session["qst6"])
    tia = int(request.session["qst7"])
    msyndrome = int(request.session["qst8"])
    atherosclerosis = int(request.session["qst9"])
    af = int(request.session["qst10"])
    lvh = int(request.session["qst11"])
    diabetes = int(request.session["qst12"])
    smoking = int(request.session["qst13"])
    sex = int(request.session["sex"])
    age = int(request.session["age"])
    # Load model
    model = pickle.load(open('model.pkl', 'rb'))
    # Make predictions
    pred = model.predict([
        [history, hypertension, inactivity, cardiovascular, hyperlidermia, alcohol, tia, msyndrome, atherosclerosis,
         sex, age, af,
         lvh, diabetes, smoking]])
    probability = model.predict_proba([
        [history, hypertension, inactivity, cardiovascular, hyperlidermia, alcohol, tia, msyndrome, atherosclerosis,
         sex, age, af,
         lvh, diabetes, smoking]])
    pred = round(pred[0])
    prob = probability[0][1]
    prob = float(prob * 100)
    prob = round(prob, 2)

    result = "False"
    if prob < 50:
        result = "True"
    else:
        result = "False"
    print('result', result)
    phone = request.session['phone']
    uname = UserDet.objects.values('name').get(phone=phone)
    name = uname.get('name')
    chance_stroke = ''
    chance_stroke += "As a stroke counselor, I would like to provide recommendations for " + name + " " + str(
        age) + " years old patient with a probability of having a stroke of " + str(prob) + "%. " + name + " is "
    if sex == 1:
        chance_stroke += 'a Male '
    if sex == 0:
        chance_stroke += 'a Female '
    if history == 1:
        chance_stroke += ' with a family history of stroke,'
    if history == 0:
        chance_stroke += ' with no family history of stroke,'
    if history == 2:
        chance_stroke += ' with an unsure family history of stroke,'
    if hypertension == 1:
        chance_stroke += ' hypertensive,'
    if hypertension == 0:
        chance_stroke += ' not hypertensive,'
    if inactivity == 0:
        chance_stroke += ' not exercising.'
    if inactivity == 1:
        chance_stroke += ' exercises regularly.'
    if cardiovascular == 0:
        chance_stroke += ' has not been diagnosed of cardiovascular disease.'
    if cardiovascular == 1:
        chance_stroke += ' has been diagnosed of cardiovascular disease.'
    if hyperlidermia == 0:
        chance_stroke += 'Additionally, ' + name + ' doesn\'t have hyperlipidemia,'
    if hyperlidermia == 1:
        chance_stroke += 'Additionally, ' + name + ' suffers from hyperlipidemia,'
    if alcohol == 0:
        chance_stroke += ' doesn\'t consume alcohol,'
    if alcohol == 1:
        chance_stroke += ' consumes alcohol,'
    if tia == 0:
        chance_stroke += ' has no history of Transient Ischemic Stroke(TIA), '
    if tia == 1:
        chance_stroke += ' has a history of Transient Ischemic Stroke(TIA), '
    if msyndrome == 0:
        chance_stroke += ' not diagnosed of metabolic Syndrome, '
    if msyndrome == 1:
        chance_stroke += ' suffers from metabolic syndrome, '
    if atherosclerosis == 0:
        chance_stroke += ' does not suffer from atherosclerosis, '
    if atherosclerosis == 1:
        chance_stroke += ' suffers from atherosclerosis, '
    if af == 0:
        chance_stroke += ' has no case of atrial fibrillation, '
    if af == 1:
        chance_stroke += ' reported to have atrial fibrillation, '
    if lvh == 0:
        chance_stroke += ' does not suffer from Left Ventricular Hypertrophy, '
    if lvh == 1:
        chance_stroke += ' suffers from Left Ventricular Hypertrophy, '
    if diabetes == 0:
        chance_stroke += ' is not diabetic. '
    if diabetes == 1:
        chance_stroke += ' is diabetic. '
    if smoking == 0:
        chance_stroke += 'Finally, ' + name + ' doesn\'t smoke.'
    if smoking == 1:
        chance_stroke += 'Finally, ' + name + ' is a smoker.'
        chance_stroke += " Could you please provide detailed diet recommendations and any other advice that would help " \
                         "manage the risk factors associated with stroke in " + name + "\'s case? "
    try:
        counseling_response = ""
        response = openai.Completion.create(
            # model name used here is text-davinci-003
            # there are many other models available under the
            # umbrella of GPT-3
            model="text-davinci-003",
            # passing the user input
            prompt=chance_stroke,
            # generated output can have "max_tokens" number of tokens
            max_tokens=2000,
            temperature=0.5,
            # number of outputs generated in one call
            n=5
        )
        # creating a list to store all the outputs
        # output = ""
        # for k in response['choices']:
        # output += k['text'].strip()
        counseling_response = response["choices"][0]["text"]

        newdata = Data.objects.create(did=id, history=history, hypertension=hypertension, inactivity=inactivity,
                                      cardiovascular=cardiovascular, hyperlidermia=hyperlidermia, alcohol=alcohol,
                                      tia=tia,
                                      msyndrome=msyndrome, atherosclerosis=atherosclerosis,
                                      sex=sex, age=age, af=af, lvh=lvh, diabetes=diabetes, smoking=smoking, stroke=pred,
                                      percent=prob,
                                      phone_id=phone, advice=counseling_response)
        newdata.save()
        return redirect('results', conseling=counseling_response, pred=pred, phone=phone, prob=prob, result=result,
                        id=id)
    except ConnectionError as e:
        counseling_response = "Network Error"
        messages.info(request, counseling_response)
        return render(request, 'message.html')
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (including DNS resolution) from the requests library
        error_message = "Network Error: " + str(e)
        messages.info(request, error_message)
        return render(request, 'message.html')
    except openai.OpenAIError as e:
        error_message = "OpenAI API Error: " + str(e)
        messages.info(request, error_message)
        return render(request, 'message.html')
    except openai.OpenAIError as e:
        # Handle the OpenAI API key error gracefully
        error_message = "OpenAI API Key Error: " + str(e)
        messages.info(request, error_message)
        return render(request, 'message.html')


def results(request, conseling, pred, phone, prob, result, id):
    title = "Results"
    try:
        users = DocDet.objects.all()
    except DocDet.DoesNotExist:
        users = None
    return render(request, 'results.html',
                  context={"title": title, "conseling": conseling, 'pred': pred, 'users': users, 'phone': phone,
                           'prob': prob, 'result': result, "id": id})


def comp(PROMPT, MaxToken, outputs):
    # using OpenAI's Completion module that helps execute
    # any tasks involving text
    try:

        response = openai.Completion.create(
            # model name used here is text-davinci-003
            # there are many other models available under the
            # umbrella of GPT-3
            model="text-davinci-003",
            # passing the user input
            prompt=PROMPT,
            # generated output can have "max_tokens" number of tokens
            max_tokens=MaxToken,
            temperature=0.5,
            # number of outputs generated in one call
            n=outputs
        )
        # creating a list to store all the outputs
        # output = ""
        # for k in response['choices']:
        # output += k['text'].strip()
        return response["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (including DNS resolution) from the requests library
        error_message = "Network Error: " + str(e)
        return error_message
    except openai.OpenAIError as e:
        error_message = "OpenAI API Error: " + str(e)
        return error_message
    except openai.OpenAIError as e:
        # Handle the OpenAI API key error gracefully
        error_message = "OpenAI API Key Error: " + str(e)
        return error_message


def printureport(request, id):
    title = "Detail Report"
    gender = Data.objects.values('sex').get(did=id)
    sex = gender.get('sex')
    cgender = ""
    if sex == 1:
        cgender = "Male"
    elif sex == 0:
        cgender = "Female"
    cage = Data.objects.values('age').get(did=id)
    age = cage.get('age')
    chistory = Data.objects.values('history').get(did=id)
    history = chistory.get('history')
    chypertension = Data.objects.values('hypertension').get(did=id)
    hypertension = chypertension.get('hypertension')
    cinactivity = Data.objects.values('inactivity').get(did=id)
    inactivity = cinactivity.get('inactivity')
    ccardiovascular = Data.objects.values('cardiovascular').get(did=id)
    cardiovascular = ccardiovascular.get('cardiovascular')
    chyperlidermia = Data.objects.values('hyperlidermia').get(did=id)
    hyperlidermia = chyperlidermia.get('hyperlidermia')
    calcohol = Data.objects.values('alcohol').get(did=id)
    alcohol = calcohol.get('alcohol')
    ctia = Data.objects.values('tia').get(did=id)
    tia = ctia.get('tia')
    cmsyndrome = Data.objects.values('msyndrome').get(did=id)
    msyndrome = cmsyndrome.get('msyndrome')

    catherosclerosis = Data.objects.values('atherosclerosis').get(did=id)
    atherosclerosis = catherosclerosis.get('atherosclerosis')
    caf = Data.objects.values('af').get(did=id)
    af = caf.get('af')
    clvh = Data.objects.values('lvh').get(did=id)
    lvh = clvh.get('lvh')
    cdiabetes = Data.objects.values('diabetes').get(did=id)
    diabetes = cdiabetes.get('diabetes')
    csmoking = Data.objects.values('smoking').get(did=id)
    smoking = csmoking.get('smoking')
    cstroke = Data.objects.values('stroke').get(did=id)
    stroke = cstroke.get('stroke')
    cadvice = Data.objects.values('advice').get(did=id)
    advice = cadvice.get('advice')
    cphone = Data.objects.values('phone_id').get(did=id)
    phone = cphone.get('phone_id')
    model = pickle.load(open('model.pkl', 'rb'))
    probability = model.predict_proba([
        [history, hypertension, inactivity, cardiovascular, hyperlidermia, alcohol, tia, msyndrome, atherosclerosis,
         sex, age, af,
         lvh, diabetes, smoking]])
    prob = probability[0][1]
    prob = float(prob * 100)
    prob = round(prob, 2)
    # result = prob < 70
    return render(request, 'printureport.html',
                  context={'title': title, "sex": sex, "age": age, "history": history, "hypertension": hypertension,
                           "inactivity": inactivity, "cardiovascular": cardiovascular,
                           "hyperlidermia": hyperlidermia, "alcohol": alcohol, "tia": tia, "msyndrome": msyndrome,
                           "atherosclerosis": atherosclerosis, "af": af, "lvh": lvh,
                           "diabetes": diabetes, "smoking": smoking, "stroke": stroke, "phone": phone,
                           "advice": advice, "cgender": cgender, 'prob': prob})


def requestdataset(request):
    title = 'New Researcher'
    # users = User.objects.filter(is_superuser=False)
    context = {'title': title}
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        u = Researchers.objects.create(email=email, name=name, date=timezone.now())
        u.save()
        return redirect(reverse('dataset'))
    else:
        return render(request, 'requestdataset.html', context)


def viewallreports(request):
    title = "View Reports"
    reports = Data.objects.all()
    context = {'title': title, 'reports': reports}
    return render(request, "viewallreports.html", context)
