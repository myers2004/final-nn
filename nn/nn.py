# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        #Calculate the linear transformed matrix for the current layer
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        #Specify allowed activation functions
        activation_functions = ['sigmoid', 'relu']

        #Erro handling if an unknown activation fucntion is specified
        if activation not in activation_functions:
            raise(KeyError('Invalid Activation Fucntion: ' + activation))
        

        #Apply the activation function
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            A_curr = self._relu(Z_curr)

        return Z_curr, A_curr


    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        #Start with the inputs
        A = X
        cache = {}

        #Save the input layer to cache
        cache['A0'] = A

        num_layers = len(self.arch)

        #Do a single foward pass for every layer
        for i in range(1, num_layers + 1):
            #Get layer values
            A_prev = A
            activation = self.arch[i-1]['activation']
            W_curr = self._param_dict['W' + str(i)]
            b_curr = self._param_dict['b' + str(i)]

            #Do a single foward for layer i
            Z, A = self._single_forward(W_curr, b_curr, A_prev, activation)

            #save to cache
            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A

        #Setthe output to the last activation layer
        output = A
        
        return output, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #Specify allowed activation functions
        activation_functions = ['sigmoid', 'relu']

        #Erro handling if an unknown activation fucntion is specified
        if activation_curr not in activation_functions:
            raise(KeyError('Invalid Activation Fucntion: ' + activation_curr))

        #Get dZ_curr for the layer
        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        #Calculate the dW and dB for the current layer and dA for the previos
        dW_curr = np.dot(dZ_curr, A_prev.T)
        db_curr = dZ_curr
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        #Define allowed loss functions
        loss_functions = ['BCE' ,'MSE']
        if self._loss_func not in loss_functions:
            raise(KeyError('Unknown loss function:' + self._loss_func))

        grad_dict = {}

        num_layers = int(len(cache) / 2)

        #Need to first get the error fucntion gradient for the output layer
        output = cache['A' + str(num_layers)]

        if self._loss_func == 'BSE':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        #Now we backpropagate to get the rest
        for i in range(num_layers, 0, -1):

            #Get parameters for current layer i (and the previous activation layer)
            activation = self.arch[i-1]['activation']
            W_curr = self._param_dict['W' + str(i)]
            b_curr = self._param_dict['b' + str(i)]
            Z_curr = cache['Z' + str(i)]
            A_prev = cache['A' + str(i-1)]

            #do a single backprop for the current layer i
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

            #Save to grad_dict
            grad_dict['dA' + str(i-1)] = dA_prev
            grad_dict['dW' + str(i)] = dW_curr
            grad_dict['db' + str(i)] = db_curr

            #set dA_prev to dA_curr for the next layer
            dA_curr = dA_prev

        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        num_layers = len(self.arch)
        m = self._batch_size

        #Update W and b for each layer
        for i in range(1, num_layers + 1):
            self._param_dict['W' + str(i)] = self._param_dict['W' + str(i)] - self._lr * (1/m * grad_dict['dW' + str(i)])

            self._param_dict['b' + str(i)] = self._param_dict['b' + str(i)] - self._lr * (1/m * grad_dict['db' + str(i)])

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        #Check the correct batch size is specified
        if np.shape(X_train)[1] != self._batch_size:
            raise(IndexError('Input data does not match specified batch size'))

        #Check for valid loss function
        loss_functions = ['BCE' ,'MSE']
        if self._loss_func not in loss_functions:
            raise(KeyError('Unknown loss function:' + self._loss_func))

        #Train!
        i = 0
        
        while i < self._epochs:

            #Get the prediction for the training and validation set
            y_hat_train, cache_train = self.forward(X_train)
            y_hat_val, cache_val = self.forward(X_val)

            #Calculate the loss for the training and validation set
            if self._loss_func == 'BSE':

                loss_train = self._binary_cross_entropy(y_train, y_hat_train)
                loss_val = self._binary_cross_entropy(y_train, y_hat_val)
            
            else:

                loss_train = self._mean_squared_error(y_train, y_hat_train)
                loss_val = self._mean_squared_error(y_train, y_hat_val)

            #Backpropagate the training set and use it to update weights and bias
            grad_dict_train = self.backprop(y_train, y_hat_train, cache_train)
            self._update_params(grad_dict_train)

            #track the loss per epoch
            per_epoch_loss_train.append(loss_train)
            per_epoch_loss_val.append(loss_val)

            i = i + 1

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        y_hat, cache = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))

        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        A = self._sigmoid(Z)

        dZ = A * (1 - A) * dA

        return dZ


    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0,Z)

        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        A = self._relu(Z)

        dZ = (A > 0) * 1 * dA

        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        num_pred = len(y_hat)

        #Calculate loss using the binary cross entropy loss equation in two steps
        # Done in the same way as loggistic regression
        loss = -(np.dot(y,np.log(y_hat)) + np.dot(1 - y, np.log(1 - y_hat)))
        loss = loss / num_pred

        return loss



    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        num_pred = len(y_hat)

        dA = -1/num_pred * (y/y_hat - (1 - y)/(1 - y_hat))

        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        num_pred = len(y_hat)

        loss = (y - y_hat)**2 / num_pred

        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        num_pred = len(y_hat)

        dA = -2/num_pred * (y - y_hat)

        return dA