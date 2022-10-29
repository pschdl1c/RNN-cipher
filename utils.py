import numpy as np
import csv


def create_key(block_size: int, length: int) -> tuple:
    """Creates random key.
    
    Args:
    block_size : integer
        The size of the block to create the key.
        Must be equal to the size of the output block RNN-Cipher.
    length : integer
        Key length.

    Return:
        tuple np.array (x, y) : tuple with arrays, shape x (length, block_size)
                                                   shape y (length, block_size)
    """

    x_train = np.array(list(np.random.bytes(length * block_size)))
    y_train = np.array(list(np.random.bytes(length * block_size)))

    x_train = np.reshape(x_train, (length, block_size)) / 256
    y_train = np.reshape(y_train, (length, block_size)) / 256

    return (x_train, y_train)


def save_key(dataset: tuple, path_to_file: str) -> None:
    """Saves the tuple with the key in csv.
    
    Args:
    dataset : tuple
        Tuple with x, y.
    path_to_file : string
        The path to save the file.

    Return:
        None.
    """

    x_train, y_train = dataset
    with open(path_to_file, 'w', newline='') as f:
        writer = csv.writer(f)       
        
        for x in x_train:
            writer.writerow(x)      
        f.write('\n')      
        for y in y_train:
            writer.writerow(y)
    

def load_key(path_to_file: str) -> tuple:
    """Loading the key from the path.
    
    Args:
    path_to_file : string
        The path where the file is located.

    Return:
        tuple np.array (x, y) : tuple with arrays, shape x (length, block_size)
                                                   shape y (length, block_size)
    """

    x_train = []
    y_train = []
    
    with open(path_to_file, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                break
            x_train.append([float(i) for i in row])
        for row in reader:
            y_train.append([float(i) for i in row])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return (x_train, y_train)


def pad(data_to_pad: bytes, block_size: int) -> bytes:
    """Apply standard padding.
    
    Args:
    data_to_pad : byte string
        The data that needs to be padded.
    block_size : integer
        The block boundary to use for padding.

    Return:
        byte string : the original data with the appropriate padding added at the end.
    """

    pad_len = block_size - len(data_to_pad) % block_size
    padding = (chr(pad_len)*pad_len).encode('utf-8')

    return data_to_pad + padding


def unpad(padded_data: bytes, block_size: int) -> bytes:
    """Remove standard padding.
    
    Args:
    padded_data : byte string
        A piece of data with padding that needs to be stripped.
    block_size : integer
        The block boundary to use for padding.

    Return:
        byte string : data without padding.
    """
    
    pdata_len = len(padded_data)
    padding_len = padded_data[-1]

    if padding_len < 1 or padding_len > min(block_size, pdata_len):
        raise ValueError("Padding is incorrect.")
    if pdata_len % block_size:
        raise ValueError("Input data is not padded")
        
    return padded_data[:-padding_len]