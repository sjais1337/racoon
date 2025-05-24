from django import forms
from .models import ClientUpdateSubmission

class ClientUpdateUploadForm(forms.ModelForm):
    class Meta:
        model = ClientUpdateSubmission
        fields = ['weights_file'] 
        widgets = {
            'weights_file': forms.ClearableFileInput(attrs={'accept': '.pth'}),
        }

    def clean_weights_file(self):
        file = self.cleaned_data.get('weights_file', False)
        if file:
            if not file.name.endswith('.pth'):
                raise forms.ValidationError("Only .pth files are allowed.")
        else:
            raise forms.ValidationError("Couldn't read uploaded file.")
        return file