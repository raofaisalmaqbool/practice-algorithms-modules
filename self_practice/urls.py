"""self_practice URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the includes() function: from django.urls import includes, path
    2. Add a URL to urlpatterns:  path('blog/', includes('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.portfolio_home, name='portfolio_home'),  # New portfolio landing page
    path('base/', views.base, name='base'),
    path('home/', views.home, name='home'),
    path('courses/', views.courses, name='courses'),
    path('index/', views.index, name='index'),
    path('contact_us/', views.contact_us, name='contact_us'),
    path('about_us/', views.about_us, name='about_us'),
    path('courses/<productid>', views.coursedetails, name='coursedetails'),
    path('submitform/', views.submitform, name='submitform'),
    path('calculator/', views.calculator, name='calculator'),
    path('even_odd/', views.even_odd, name='even_odd'),
    path('newsdetails/<slug>', views.newsdetails, name='newsdetails'),
    # New demo URLs for algorithms and ML
    path('algorithms/', views.algorithms_demo, name='algorithms_demo'),
    path('data-structures/', views.data_structures_demo, name='data_structures_demo'),
    path('ml-demo/', views.ml_demo, name='ml_demo'),

]
