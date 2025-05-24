from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from fl_platform.models import TrainingGroup

INITIAL_TRAINING_GROUPS_DATA = [
    {
        "name": "Adult Income Prediction",
        "description": "Federated learning model to predict if income is >50K based on census data.",
        "model_config_key": "ADULT_INCOME_LOGISTIC_REGRESSION"
    },
    {
        "name": "Smoking Prediction Bio Signals",
        "description": "Predicts smoking status based on bio-signal data.",
        "model_config_key": "SMOKING_BIO_LOGISTIC_REGRESSION"
    },
    {
        "name": "Lumpy Skin Disease Prediction",
        "description": "Predicts Lumpy Skin Disease occurrence.",
        "model_config_key": "LUMPY_SKIN_LOGISTIC_REGRESSION"
    },
    {
        "name": "Credit Score Classification",
        "description": "Classifies credit scores (e.g., Good, Standard, Poor).",
        "model_config_key": "CREDIT_SCORE_SIMPLE_NN"
    },
]

class Command(BaseCommand):
    help = 'Creates the initial set of Training Groups in the database.'

    def handle(self, *args, **options):
        self.stdout.write("Creating training groups...")

        for group_data in INITIAL_TRAINING_GROUPS_DATA:
            obj, created = TrainingGroup.objects.get_or_create(
                name=group_data["name"],
                defaults={
                    "description": group_data["description"],
                    "model_config_key": group_data["model_config_key"]
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Successfully created TrainingGroup: {obj.name}"))
            else:
                updated_fields = False
                if obj.description != group_data["description"]:
                    obj.description = group_data["description"]
                    updated_fields = True
                if obj.model_config_key != group_data["model_config_key"]:
                    obj.model_config_key = group_data["model_config_key"]
                    updated_fields = True
                
                if updated_fields:
                    obj.save()
                    self.stdout.write(self.style.SUCCESS(f"Successfully updated existing TrainingGroup: {obj.name}"))
                else:
                    self.stdout.write(f"TrainingGroup '{obj.name}' already exists and is up-to-date.")

        self.stdout.write(self.style.SUCCESS("Finished processing initial training groups."))
