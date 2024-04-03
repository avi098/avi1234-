import requests

projectId = "2V4bMRauqdIE51S69oLg0UgsKr3"
projectSecret = "7dd9f75ecef30468205325ea48eac148"
endpoint = "https://ipfs.infura.io:5001"


def add_text_to_ipfs(text):
    files = {
        'content': text.encode('utf-8')
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
