# Generated by Django 5.2.1 on 2025-05-18 17:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fl_platform', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='traininggroup',
            name='model_architecture_type',
        ),
        migrations.AddField(
            model_name='traininggroup',
            name='model_config_key',
            field=models.CharField(default='ADULT_INCOME_LOGISTIC_REGRESSION', help_text='Key to look up model params in config.', max_length=100),
        ),
    ]
