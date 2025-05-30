# Generated by Django 5.2.1 on 2025-05-18 16:43

import django.db.models.deletion
import fl_platform.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TrainingGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('description', models.TextField(blank=True)),
                ('model_architecture_type', models.CharField(default='LogistcRegression', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='GlobalModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('round_number', models.PositiveBigIntegerField(default=0)),
                ('model_weights_file', models.FileField(blank=True, null=True, upload_to=fl_platform.models.model_upload_path)),
                ('accuracy', models.FloatField(blank=True, null=True)),
                ('loss', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('is_active', models.BooleanField(default=True)),
                ('training_group', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='global_models', to='fl_platform.traininggroup')),
            ],
            options={
                'ordering': ['-round_number'],
                'unique_together': {('training_group', 'round_number')},
            },
        ),
    ]
