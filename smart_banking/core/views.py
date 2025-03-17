
import os
from django.conf import settings
from django.shortcuts import redirect

VERCEL_FRONTEND_URL = "https://nischhalshrestha.com.np"


# Create your views here.
def index(request):
    return redirect(VERCEL_FRONTEND_URL)