from django.shortcuts import render, redirect
from django.http import HttpResponse
import requests
from .models import *
from .forms import SignUpForm, LogInForm
from .models import UserProfile

projectId = "f9b13dd46b5a455daff2e669a6643ee8"
projectSecret = "dWVe2B7tl0Bsqna23s1nJNEsfxXZJrJUHxYLB+MOh91z7NsvUfIIGw"
endpoint = "https://ipfs.infura.io:5001"


def add_text_to_ipfs(text):
    files = {
        'content': text
    }
    response1 = requests.post(endpoint + '/api/v0/add', files=files, auth=(projectId, projectSecret))
    hash = response1.text.split(",")[1].split(":")[1].replace('"', '')
    return hash


def get_text_from_ipfs(ipfs_hash):
    params = {
        'arg': ipfs_hash
    }
    response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))
    text = response2.text
    return text


# Django view to demonstrate encryption and decryption
def register(request):
    if request.method == "POST":
        # Get data from the HTML form
        data1 = request.POST.get("data1", "")
        data2 = request.POST.get("data2", "")
        data3 = request.POST.get("data3", "")
        data4 = request.POST.get("data4", "")
        data5 = request.POST.get("data5", "")

        # Save data to IPFS
        ipfs_hash_1 = add_text_to_ipfs(data1)
        ipfs_hash_2 = add_text_to_ipfs(data2)
        ipfs_hash_3 = add_text_to_ipfs(data3)
        ipfs_hash_4 = add_text_to_ipfs(data4)
        ipfs_hash_5 = add_text_to_ipfs(data5)

        # Save the IPFS hashes to the database
        encrypted_data_entry = datas(
            data1=ipfs_hash_1,
            data2=ipfs_hash_2,
            data3=ipfs_hash_3,
            data4=ipfs_hash_4,
            data5=ipfs_hash_5,
        )
        encrypted_data_entry.save()

        # Redirect to a success page or display a message
        return redirect('decrypt_all_data')

    # Render the HTML form if it's a GET request
    return render(request, "form.html")


def decrypt_all_data(request):
    # Retrieve all encrypted data entries from the database
    encrypted_data_entries = datas.objects.all()

    # Create a list to store decrypted data
    decrypted_data_list = []

    # Iterate through each encrypted entry
    for entry in encrypted_data_entries:
        data1 = get_text_from_ipfs(entry.data1)
        data2 = get_text_from_ipfs(entry.data2)
        data3 = get_text_from_ipfs(entry.data3)
        data4 = get_text_from_ipfs(entry.data4)
        data5 = get_text_from_ipfs(entry.data5)

        # Add the retrieved data to the list
        decrypted_data_list.append({
            'data1': data1,
            'data2': data2,
            'data3': data3,
            'data4': data4,
            'data5': data5
        })

    # Render a template to display the decrypted data
    return render(request, 'decrypted_data.html', {'decrypted_data_list': decrypted_data_list})



def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


def login(request):
    if request.method == 'POST':
        form = LogInForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = UserProfile.objects.filter(username=username, password=password).first()
            if user:
                # Authentication successful, redirect to dashboard or home page
                return redirect('register')
            else:
                # Authentication failed
                error = "Invalid username or password."
                return render(request, 'login.html', {'form': form, 'error': error})
    else:
        form = LogInForm()
    return render(request, 'login.html', {'form': form})


def encryption_demo(request):
    return render(request,'encryption_demo.html')