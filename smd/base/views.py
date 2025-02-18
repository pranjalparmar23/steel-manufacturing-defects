from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect

@login_required
def home(request):
    return render(request, "home.html", {})

def authView(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST or None)
        if form.is_valid():
            form.save()
            return redirect("base:login")
    else:
        form = UserCreationForm()
    return render(request, "registration/signup.html", {"form": form})

def about(request):
    return render(request, "about.html", {})

def contact(request):
    return render(request, "contact.html", {})

def services(request):
    return render(request, "services.html", {})

def footer(request):
    return render(request, "footer.html", {})

def upload(request):
    return render(request, "upload.html", {})

def detect(request):
    return render(request, "detect.html", {})




