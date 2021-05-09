from django.shortcuts import render
from . import hope


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
# Create your views here.

def first(request):
    return render(request,'index.html')
@csrf_exempt
def second(request):
    if request.method=='POST':
        hope.Detector()
    return render(request,'index.html')
    