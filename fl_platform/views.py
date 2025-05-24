from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import  login_required
from .models import TrainingGroup, GlobalModel, ClientUpdateSubmission
from .forms import ClientUpdateUploadForm
from django.contrib.auth.decorators import user_passes_test
from django.core.files.base import ContentFile
from django.conf import settings
import torch
import io

from django.contrib import messages
from .models_pytorch import LogisticRegressionModel, SimpleNN #
from django.http import FileResponse, Http404, HttpResponse 
from django.templatetags.static import static 

from .fl_utils import get_pytorch_model_instance, server_aggregate_weights_util


def is_admin_user(user):
    return user.is_authenticated and user.is_superuser


@login_required
def list_training_groups(request):
    
    groups = TrainingGroup.objects.all().order_by('name')
    
    context = {
        'training_groups': groups,
    }

    return render(request, 'fl_platform/training_group_list.html', context)

@login_required
def training_group_detail(request, group_id):
    group = get_object_or_404(TrainingGroup, pk=group_id)

    active_global_model = GlobalModel.objects.filter(training_group=group, is_active=True).first()

    historical_global_models = GlobalModel.objects.filter(training_group=group).order_by('-round_number')

    total_submissions = 0
    unique_submitters = 0
    upload_form = ClientUpdateUploadForm()

    current_model_config = settings.FL_MODEL_CONFIGS.get(group.model_config_key, {})

    if active_global_model:
        client_updates_qs = active_global_model.client_updates.all()
        total_submissions = client_updates_qs.count()
        unique_submitters = client_updates_qs.values('submitted_by').distinct().count()

    context = {
        'group': group,
        'active_global_model': active_global_model,
        'historical_global_model': historical_global_models,
        'total_submissions': total_submissions,
        'unique_submitters': unique_submitters,
        'upload_form': upload_form,
        'current_model_config': current_model_config
    }

    return render(request, 'fl_platform/training_group_detail.html', context)

@user_passes_test(is_admin_user)
def initialize_global_model(request, group_id):
    group = get_object_or_404(TrainingGroup, pk=group_id)

    if GlobalModel.objects.filter(training_group=group, round_number=0).exists():
        messages.warning(request, f"Round 0 model already exists for group '{group.name}'.")
        return redirect('training_group_detail', group_id=group.id)
    
    model_config_key = group.model_config_key
    config = settings.FL_MODEL_CONFIGS[model_config_key]

    try:
        pytorch_model = get_pytorch_model_instance(model_config_key)
        pytorch_model.eval()
    except Exception as e:
        messages.error(request, "Error instantiating PyTorch model.")

    try:
        weights_buffer = io.BytesIO()
        torch.save(pytorch_model.state_dict(), weights_buffer)
        weights_buffer.seek(0)

        filename = f"initial_group_{group.id}_round_0.pth"
        gm = GlobalModel(
            training_group=group,
            round_number=0,
            is_active=True
        )
        gm.model_weights_file.save(filename, ContentFile(weights_buffer.read()), save=True)

        messages.success(request, "Successfully initialized Round 0 model.")
    except Exception as e:
        print(e)
        messages.error(request, "Error saving global model.")
        return redirect('training_group_detail', group_id=group.id)

    return redirect('training_group_detail', group_id=group.id)

@user_passes_test(is_admin_user)
def trigger_aggregation(request, group_id, model_id):
    group = get_object_or_404(TrainingGroup, pk=group_id)
    source_global_model = get_object_or_404(GlobalModel, pk=model_id)

    if request.method == 'POST':
        submissions = ClientUpdateSubmission.objects.filter(
            training_group=group,
            global_model=source_global_model
        ).order_by('submitted_by_id','-uploaded_at')

        to_process = []
        processed_users = set()
        for submission in submissions:
            if submission.submitted_by_id not in processed_users:
                if submission.weights_file:
                    to_process.append(submission)
                    processed_users.add(submission.submitted_by)
        
        if not to_process:
            messages.warning(request, "No updates found to aggregate.")
            return redirect('training_group_detail', group_id=group.id)
        
        client_state_dicts = []

        for submission in to_process:
            try:
                submission.weights_file.open('rb')
                state_dict = torch.load(submission.weights_file, map_location=torch.device('cpu'))
                client_state_dicts.append(state_dict)
                submission.weights_file.close()
            except Exception as e:
                print(e)
                messages.error(request, 'Error loading weights.')
                return redirect('training_group_detail', group_id=group.id)
        
        if not client_state_dicts:
            messages.error(request, "No client weights could be loaded for aggregation.")
            return redirect('training_group_detail', group_id=group.id)

        try:
            base_model = get_pytorch_model_instance(group.model_config_key)
            base_model_structure = base_model.state_dict()
        except Exception as e:
            messages.error(request, f"Error instantiating base model for aggregation: {e}")
            return redirect('training_group_detail', group_id=group.id)
    
        new_aggregated_weights = server_aggregate_weights_util(base_model_structure, client_state_dicts)

        if new_aggregated_weights is None:
            messages.error(request, "Aggregation resulted in no new weights (maybe no client updates?).")
            return redirect('training_group_detail', group_id=group.id)

        weights_buffer = io.BytesIO()
        torch.save(new_aggregated_weights, weights_buffer)
        weights_buffer.seek(0)

        new_round_number = source_global_model.round_number + 1
        new_filename = f"aggregated_model_group_{group.id}_round_{new_round_number}.pth"

        new_global_model = GlobalModel(
            training_group=group,
            round_number=new_round_number,
            is_active=True 
        )

        new_global_model.model_weights_file.save(new_filename, ContentFile(weights_buffer.read()), save=True)

        ClientUpdateSubmission.objects.filter(id__in=[s.id for s in to_process]).update(status='aggregated')

        messages.success(request, f"Successfully aggregated {len(client_state_dicts)} client updates. New Global Model for Round {new_round_number} created and activated.")
        return redirect('training_group_detail', group_id=group.id)
    else:
        messages.info(request, "Aggregation must be triggered via POST.")
        return redirect('training_group_detail', group_id=group.id)

@login_required
def download_global_model_weights(request, group_id, model_id):
    group = get_object_or_404(TrainingGroup, pk=group_id)
    global_model = get_object_or_404(GlobalModel, pk=model_id, training_group=group)

    if not global_model.model_weights_file:
        raise Http404("Model weights file not found for this global model instance.")

    try:
        return FileResponse(global_model.model_weights_file.open('rb'), as_attachment=True, filename=global_model.model_weights_file.name.split('/')[-1])
    except FileNotFoundError:
        raise Http404("Model weights file not found on disk.")
    except Exception as e:
        return HttpResponse("Error downloading file.", status=500)

@login_required
def upload_client_update(request, group_id):
    group = get_object_or_404(TrainingGroup, pk=group_id)
    active_global_model = GlobalModel.objects.filter(training_group=group, is_active=True).first()

    if not active_global_model:
        return redirect('training_group_detail', group_id=group.id)
    
    if request.method == 'POST':
        form = ClientUpdateUploadForm(request.POST, request.FILES)

        if form.is_valid():
            submission = form.save(commit=False)
            submission.training_group = group
            submission.submitted_by = request.user
            submission.global_model = active_global_model
            submission.save()

            messages.success(request, "Your model update has been uploaded.")

            return redirect('training_group_detail', group_id=group.id)
        else: 
            messages.error(request, "There was an error with your upload.")
    else:
        form = ClientUpdateUploadForm()

    context = {
        'form': form,
        'group': group,
        'active_global_model': active_global_model
    }

    return redirect('training_group_detail', group_id=group.id)
    
    
