from django.db import models
from django.conf import settings

def model_upload_path(instance, filename):
    return f'training_groups/group_{instance.training_group.id}/global_models/{filename}'

class TrainingGroup(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    model_config_key = models.CharField(max_length=100, default='ADULT_INCOME_LOGISTIC_REGRESSION', help_text='Key to look up model params in config.')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
    
class GlobalModel(models.Model):
    training_group = models.ForeignKey(TrainingGroup, related_name='global_models', on_delete=models.CASCADE)
    round_number = models.PositiveBigIntegerField(default=0)
    model_weights_file = models.FileField(upload_to=model_upload_path, null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-round_number']
        unique_together = [['training_group', 'round_number']]

    def __str__(self):
        return f"{self.training_group.name} - Round {self.round_number}"
    
    def save(self, *args, **kwargs):
        if self.is_active:
            GlobalModel.objects.filter(training_group=self.training_group, is_active=True).exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

def client_update_upload_path(instance, filename):
    round_num = instance.global_model.round_number if instance.global_model else 'unknown_round'
    user_id = instance.submitted_by.id if instance.submitted_by else 'unknown_user'
    return f'training_groups/group_{instance.training_group.id}/client_updates/round_{round_num}/user_{user_id}_{filename}'

class ClientUpdateSubmission(models.Model):
    training_group = models.ForeignKey(TrainingGroup, on_delete=models.CASCADE, related_name='client_submissions')
    global_model = models.ForeignKey(GlobalModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='client_updates')
    submitted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    weights_file = models.FileField(upload_to=client_update_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # TODO: Potentially add a status field: 'pending_aggregation', 'aggregated', 'error'
    status = models.CharField(max_length=20, default='pending_aggregation') 

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        round_num = self.global_model.round_number
        return f"Update by {self.submitted_by.username} for {self.training_group.name} (Based on Round {round_num})"
    

