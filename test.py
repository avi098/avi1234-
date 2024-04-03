from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def encrypt_and_store_data():
    key = b'\xf4\xaaxz\xe3\xec\xda\xd5\x1e\xcfZ\x9bA\x8f\x15\xdd\xb2\x085\x04\x07\x8e\xf4\xaa\x7f\x04\xb2arun\xa2'
    data = "arun"
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext = cipher.encrypt(data)

    encrypted_data = ciphertext
    print(encrypted_data)
    return encrypted_data

    # You should also store the key securely if you need to decrypt it later.
    # Consider using a key management solution.


def retrieve_and_decrypt_data():
    encrypted_data_obj = "a"
    if encrypted_data_obj:
        key = b'\xf4\xaaxz\xe3\xec\xda\xd5\x1e\xcfZ\x9bA\x8f\x15\xdd\xb2\x085\x04\x07\x8e\xf4\xaa\x7f\x04\xb2arun\xa2'
        cipher = AES.new(key, AES.MODE_EAX, nonce=encrypted_data_obj.nonce)
        decrypted_data = cipher.decrypt(encrypted_data_obj)
        return decrypted_data.decode('utf-8')
    else:
        return None


print(encrypt_and_store_data())
