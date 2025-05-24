from django.contrib import admin
from .models import TrainingGroup, GlobalModel, ClientUpdateSubmission

@admin.register(TrainingGroup)
class TrainingGroupAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at', 'updated_at')
    search_fields = ('name',)

@admin.register(GlobalModel)
class GlobalModeAdmin(admin.ModelAdmin):
    list_display = ('training_group', 'round_number','model_weights_file', 'accuracy', 'loss', 'is_active', 'created_at')
    list_filter = ('training_group', 'is_active', 'round_number')
    search_fields = ('training_group__name',)
    list_editable = ('is_active',)

@admin.register(ClientUpdateSubmission)
class ClientUpdateSubmissionAdmin(admin.ModelAdmin):
    list_display = ('training_group', 'global_model_round', 'submitted_by', 'weights_file', 'uploaded_at')
    list_filter = ('training_group', 'global_model__round_number', 'submitted_by')
    search_fields = ('submitted_by__username', 'training_group__name')
    readonly_fields = ('uploaded_at',)

    def global_model_round(self, obj):
        if obj.global_model:
            return obj.global_model.round_number
        return "N/A"
    global_model_round.short_description = 'Based on Round #'