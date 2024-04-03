from django import forms

class IPFSTextForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea)
