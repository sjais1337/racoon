from django.urls import path
from . import views

urlpatterns = [
    path('', views.list_training_groups, name='list_training_groups'),
    path('<int:group_id>/', views.training_group_detail, name='training_group_detail'),
    path('<int:group_id>/initialize/', views.initialize_global_model, name='initialize_global_model'),
    path('<int:group_id>/models/<int:model_id>/download_weights/', views.download_global_model_weights, name='download_global_model_weights'),
    path('<int:group_id>/upload_update/', views.upload_client_update, name='upload_client_update'),
    path('<int:group_id>/models/<int:model_id>/trigger_aggregation/', views.trigger_aggregation, name='trigger_aggregation'),
]