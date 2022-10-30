# Recurrent neural networks cipher
This is the implementation of encryption using recurrent neural networks.
### *Installation:*
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is the recommended approach for installing TensorFlow with GPU support.

Copy this commands to your console:  
`$ git clone https://github.com/pschdl1c/RNN-cipher.git`  
`$ cd RNN-cipher`  
`$ conda create --name <env> --file requirements.txt`  

Check `run.py` file.  

For Windows users: [click here](https://stackoverflow.com/a/72256408/20294758) if there are problems with XLA.

### How it works:
![RNN](https://github.com/pschdl1c/RNN-cipher/raw/main/pic/pic.PNG)
This RNN has a multilayer structure with two limitations:
1) the dimension of the input vector X is twice the dimension of the output vector Y;
2) one of the hidden layers has only one neuron (ξ).

Symmetric encryption is performed in two stages: key expansion and data encryption/decryption.


### *Note:*
This is an experimental algorithm, do not use it in real problems :)