from django.urls import path
from .views import HomePageView, AboutPageView, predictImage
# from .views import homePageView

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('predictImage/', predictImage, name="predict")
]

"""
- regex of the name of the url
- homePageView: the name of the function that the url will show
- name: optional, the name of your view


urlpatterns = [
    path('', homePageView, name='home')
]
"""
