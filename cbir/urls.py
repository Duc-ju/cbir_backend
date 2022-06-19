from django.urls import path
from . import views
urlpatterns = [
    path('<image>/', views.QueryView.as_view()),
    path('', views.queryInput)
]
