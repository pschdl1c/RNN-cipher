import time
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

from RNN_ciper import RNN_Cipher
from utils import create_key, load_key, save_key, pad, unpad


BLOCK_SIZE = 8
SAMPLE_LEN = 2000

if __name__ == '__main__':
    data = b'Secret data!'
    data = pad(data, BLOCK_SIZE)

    x_train, y_train = create_key(BLOCK_SIZE, SAMPLE_LEN)
    save_key((x_train, y_train), './key.csv')

    cipher = RNN_Cipher(use_saved_weights=False)
    start = time.perf_counter() 
    cipher.KeyExpansion(x_train, y_train)
    end = time.perf_counter()
    print(end - start)
    # encrypted_data = cipher.Encrypt(data)
    
    # de_cipher = RNN_Cipher(use_saved_weights=False)
    # de_cipher.KeyExpansion(x_train, y_train)
    # decrypted_data = de_cipher.Decrypt(encrypted_data)

    # print(f'Data: ', unpad(decrypted_data, BLOCK_SIZE))
    # assert data == decrypted_data

