import sys

import tensorflow as tf
import numpy as np


class RNN_Cipher:
    """ Recurrent neural network cipher.

    Parameters
    ------------
    layers_config : list (default: [16, 16, 1, 8])
        Number of neurons for each layer. The list should contain one layer with the value 1.
        The size of the output layer should be half the size of the input one.
    learning_rate : float (default: 1.0)
        The learning rate coefficient is a hyperparameter that determines the order of how weights
        will be adjusted taking into account the loss function during gradient descent.
    epochs : int (default: 4)
        The number of passes through the training set (for KeyExpansion).
    critical_rate : float (default: 0.04)
        Critical value for a self-adaptive learning procedure.
    use_saved_weights : bool (default: False)
        If True, then there should be a saved file with weights.

    Attributes
    -----------
    {under development}

    """
    def __init__(self, layers_config: list=[16, 16, 1, 8], learning_rate: float=1.0,
                 epoches: int=4, critical_rate: float=0.04,
                 use_saved_weights: bool=False) -> None:
        # Hyperparameters
        self.epochs = epoches
        self.learning_rate = learning_rate
        self.critical_rate = critical_rate
        self.initial_weights = 0.5
        self.initial_biases = 0.0

        # Layers configuration
        self.layers_config = layers_config
        self.V_index = self.layers_config.index(1)

        # Block size
        self.block_size = self.layers_config[-1]

        # Final output from key expansion process
        self.M0 = tf.Variable([[0.0] * self.block_size], shape=(1, self.block_size), dtype=tf.float32)

        # Neural network wieghts
        self.weights, self.biases = self.__init_weights(use_saved_weights)
        
        # Optimization
        self.trainable_params = list(self.weights.values()) + list(self.biases.values())
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Initial value for learning adaptation process
        self.T_prev = 0
        
    
    def __init_weights(self, use_saved_weights):
        """ Initializing weights and bias.

        Returns
        -----------
        weights, biases : set, keys = {h1,..., v, out}
        """

        weights = {}
        biases = {}
        try:
            list_saved_weights = [key[0] for key in tf.train.list_variables("./checkpoints/check")[2:]]
            self.M0.assign(tf.train.load_variable('./checkpoints/check', tf.train.list_variables("./checkpoints/check")[0][0]))
        except Exception:
            print('[!] No such saved weights...\nWeights are initialized!')
            use_saved_weights = False
        for i in range(1, len(self.layers_config)):

            if use_saved_weights == False:
                weights_tensor = tf.Variable(tf.constant(self.initial_weights, shape=(self.layers_config[i-1], self.layers_config[i])))
                biases_tensor = tf.Variable(tf.constant(self.initial_biases, shape=(self.layers_config[i],)))
            else:
                weights_tensor = tf.Variable(tf.train.load_variable('./checkpoints/check', list_saved_weights[i-1]))
                biases_tensor = tf.Variable(tf.train.load_variable('./checkpoints/check', list_saved_weights[len(list_saved_weights)//2+i-1]))

            if i == self.V_index:
                weights['v'] = weights_tensor
                biases['v'] = biases_tensor
            elif i == len(self.layers_config) - 1:
                weights['out'] = weights_tensor
                biases['out'] = biases_tensor
            else:
                weights[f'h{i}'] = weights_tensor
                biases[f'b{i}'] = biases_tensor
        
        return weights, biases

    
    def Network(self, X):
        """ Calculates the forward propagation step """
        H = X
        for i in range(1, len(self.layers_config)):
            if i == self.V_index:
                V = tf.sigmoid(tf.matmul(H, self.weights['v']) + self.biases['v'])
                H = V
            elif i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])   
        return OUT

    
    @tf.function(jit_compile=True)
    def train_batch(self, x_batch, y_batch):
        # One iteration of learning process
        Y = self.Network(x_batch)
        loss_fn = lambda: self.loss(y_batch, self.Network(x_batch))
        self.optimizer.minimize(loss_fn, self.trainable_params)
        return Y


    def KeyExpansion(self, x_train, y_train):
        """ Performs key expansion.

        Parameters
        -----------
        x_train, y_train : array, shape = [length, block_size]
            array with key

        Returns:
        ----------
        self

        """
        print('\n[!] Starting key expansion process...\n')
        for epoch in range(self.epochs):
            sys.stdout.write(f'\rEpoch {epoch+1}/{self.epochs}')
            
            Y = np.array([0.0] * self.block_size, dtype=np.float32)

            for m, y in zip(x_train, y_train):
                # Perform concatenation of output from previous iteration and current input
                x = np.array(np.concatenate((Y, m), axis=None), dtype=np.float32)
                
                x_batch = np.array([x], dtype=np.float32)
                y_batch = np.array([y], dtype=np.float32)
                
                # Calculate feed-forward output from network
                Y = self.train_batch(x_batch, y_batch)
        
        # print('\n[*] Key expansion finished!')
        
        self.M0.assign(Y)
        checkpoint = tf.train.Checkpoint(var=self.trainable_params, M0=self.M0)
        checkpoint.write("./checkpoints/check")
        print('\n[*] Weights were saved to a file!')      
        
        return
    

    def EncryptBlock(self, X):
        """ Calculates the recurrent encryption step """
        H = X
        for i in range(1, len(self.layers_config)):
            if i == self.V_index:
                V = tf.sigmoid(tf.matmul(H, self.weights['v']) + self.biases['v'])
                H = V
            elif i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])   
        return (V, OUT)
    
    
    @tf.function(jit_compile=True)
    def encrypt_batch(self, x_batch, y_batch):
        V, OUT = self.EncryptBlock(x_batch)
        # One iteration of learning process
        loss_fn = lambda: self.loss(y_batch, self.EncryptBlock(x_batch)[1])
        self.optimizer.minimize(loss_fn, self.trainable_params)
        MSE = self.loss(y_batch, self.Network(x_batch))
        return V, OUT, MSE


    def Encrypt(self, plaintext):
        """ Encrypts data.

        Parameters
        -----------
        plaintext : byte string
            Byte string to be encrypted.

        Returns:
        ----------
        ciphertext_blocks : array
            Tuples with (V, Y) ciphertext blocks.
            V - tensor scalar (intermediate neuron output signal), shape=(1, 1)
            E - tensor (error signal), shape=(1, 8)
        """
        # Restore original learning rate
        self.DropAdaptation()

        # Prepare plaintext for encryption
        plaintext_bytes = np.array(list(plaintext))
        scaled_plaintext = plaintext_bytes / 255 # Scale bytes
        plaintext_blocks = np.reshape(scaled_plaintext, [int(len(scaled_plaintext) / self.block_size), self.block_size])
             
        ciphertext_blocks = []
        Y = self.M0[0]

        for block in plaintext_blocks:                            
            x = np.array(np.concatenate((Y, block), axis=None), dtype=np.float32)
            x_batch = np.array([x], dtype=np.float32)
            y_batch = np.array([block], dtype=np.float32)

            # Encrypt one plaintext block
            V, Y, MSE = self.encrypt_batch(x_batch, y_batch)
            E = block - Y        
            ciphertext_blocks.append((V, E))

            # Adaptation of learning process
            self.LearningAdaptation(MSE)
        
        # Restore KeyExpansion weights
        self.weights, self.biases = self.__init_weights(True)
        return ciphertext_blocks



    def DecryptBlock(self, V):
        """ Calculates the output recurrent decryption step """
        H = V
        for i in range(self.V_index + 1, len(self.layers_config)):
            if i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])
        return OUT
    

    @tf.function(jit_compile=True)
    def decrypt_batch(self, M, X):
        # One iteration of learning process
        loss_fn = lambda: self.loss(M, self.Network(X))
        self.optimizer.minimize(loss_fn, self.trainable_params)
        MSE = self.loss(M, self.Network(X))
        return MSE


    def Decrypt(self, ciphertext_blocks):
        """ Decrypts data.

        Parameters
        -----------
        ciphertext_blocks : array
            Tuples with (V, Y) ciphertext blocks.
            V - tensor scalar (intermediate neuron output signal), shape=(1, 1)
            E - tensor (error signal), shape=(1, 8)

        Returns:
        ----------
        plaintext : byte string
            Decrypted byte string.
        """
        # Restore original learning rate
        self.DropAdaptation()

        plaintext_blocks = []
        
        Y_prev = self.M0

        for block in ciphertext_blocks:
            Y = self.DecryptBlock(block[0])
            M = Y + block[1]
            plaintext_blocks.append(M[0])

            # One iteration of learning process
            X = np.array([np.concatenate((Y_prev[0], M[0]), axis=None)])
            # VV, _ = self.EncryptBlock(X)
            # print('V: ',block[0], 'V`: ',VV, block[0] == VV) ## authentication message
            MSE = self.decrypt_batch(M, X)
            
            Y_prev = Y

            # Adaptation of learning process
            self.LearningAdaptation(MSE)

        # Convert plaintext blocks to string of bytes
        plaintext_blocks = np.array(plaintext_blocks) * 255 # Unscale bytes
        plaintext_blocks = np.rint(plaintext_blocks).astype(int) # Convert bytes to int
        plaintext_bytearray = plaintext_blocks.flatten()
        plaintext = bytes(plaintext_bytearray.tolist())
        return plaintext


    def LearningAdaptation(self, MSE):
        """ The trend of the training procedure is through the control 
        of the mean square error (MSE) performance function and then adjusts the learning rate """
        delta = 0.2

        T = delta * self.T_prev + (1 - delta) * MSE
        if T <= self.critical_rate:
            new_LR = self.learning_rate * 2.0
        elif T > self.critical_rate and T > self.T_prev:
            new_LR = self.learning_rate * 0.9
        else:
            new_LR = self.learning_rate
        
        # Update learning rate for optimizer
        self.learning_rate = new_LR
        self.T_prev = T


    def DropAdaptation(self):
        """ Resets learning adaptation """
        self.T_prev = 0
        self.learning_rate = 1.0