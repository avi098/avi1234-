
from django.urls import path


from . import views

urlpatterns = [
    path('', views.encryption_demo, name='encryption_demo'),
    path('register/', views.register, name='register'),
    path('decrypted_data/', views.decrypt_all_data, name='decrypt_all_data'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    ]
