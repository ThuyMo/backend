from django.urls import path
from server.views import *

urlpatterns = [
    path('process-image', process_image),
    path('get-cmnd', save_image),
    path('login', login),
    # path('detect-text',detect_text),

]
