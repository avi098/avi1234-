from django.shortcuts import render, redirect
from .models import IPFSText
from .ipfs_utils import add_text_to_ipfs, get_text_from_ipfs


def submit_text(request):
    if request.method == 'POST':
        text = request.POST['text']

        # Use the Infura IPFS API to add the text to IPFS
        ipfs_hash = add_text_to_ipfs(text)

        # Save the IPFS hash in the database
        ipfs_text = IPFSText.objects.create(ipfs_hash=ipfs_hash)

        return redirect('view_text')

    return render(request, 'submit_text.html')


def view_text(request):
    # Retrieve the IPFS hash from the database
    ipfs_text = IPFSText.objects.first()

    if ipfs_text:
        # Use the Infura IPFS API to retrieve the text from IPFS using the hash
        text = get_text_from_ipfs(ipfs_text.ipfs_hash)
    else:
        text = "No text stored on IPFS yet."

    return render(request, 'view_text.html', {'text': text})
