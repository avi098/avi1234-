import requests

projectId = "f9b13dd46b5a455daff2e669a6643ee8"
projectSecret = "dWVe2B7tl0Bsqna23s1nJNEsfxXZJrJUHxYLB+MOh91z7NsvUfIIGw"
endpoint = "https://ipfs.infura.io:5001"


def add_text_to_ipfs(text):
    files = {
        'content': text
    }

    response1 = requests.post(endpoint + '/api/v0/add', files=files, auth=(projectId, projectSecret))
    print(response1)
    hash = response1.text.split(",")[1].split(":")[1].replace('"', '')
    print(hash)

    return hash


def get_text_from_ipfs(ipfs_hash):
    params = {
        'arg': ipfs_hash
    }
    response2 = requests.post(endpoint + '/api/v0/cat', params=params, auth=(projectId, projectSecret))

    text = response2.text

    return text
