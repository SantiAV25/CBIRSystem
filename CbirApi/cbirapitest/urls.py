from django.urls import path
from .views import SimpleAPIView, ImageFeatureExtractionView

urlpatterns = [
    path('simple/', SimpleAPIView.as_view(), name='simple-api'),
    path('extract-features/', ImageFeatureExtractionView.as_view(), name='extract-features')
]
