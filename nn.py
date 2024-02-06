
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}
    gamma_values = {}
    beta_values = {}
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def batch_norm_forward(Z, gamma, beta, epsilon=1e-7):
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde, Z_norm, mean, variance

def batch_norm_backward(dZ_tilde, Z_norm, mean, variance, gamma, beta, epsilon=1e-7):
    m = dZ_tilde.shape[1]
    
    dZ_norm = dZ_tilde * gamma
    dvariance = np.sum(dZ_norm * (Z_norm - mean) * (-0.5) * (variance + epsilon)**(-1.5), axis=1, keepdims=True)
    dmean = np.sum(dZ_norm * (-1 / np.sqrt(variance + epsilon)), axis=1, keepdims=True) + dvariance * np.sum(-2 * (Z_norm - mean), axis=1, keepdims=True) / m
    dZ = dZ_norm / np.sqrt(variance + epsilon) + dvariance * 2 * (Z_norm - mean) / m + dmean / m
    
    dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
    
    return dZ, dgamma, dbeta

def dropout_forward(A, keep_prob):
    dropout_mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
    A_dropout = A * dropout_mask
    return A_dropout, dropout_mask

def dropout_backward(dA_dropout, dropout_mask):
    dA = dA_dropout * dropout_mask
    return dA
    
def single_layer_forward_propagation(A_prev, W_curr, b_curr, gamma, beta, activation="relu",  dropout_prob=0.0):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    Z_tilde, Z_norm, mean, variance = batch_norm_forward(Z_curr, gamma, beta)
    A_curr = activation_func(Z_tilde)

    if dropout_prob > 0.0:
        A_curr, dropout_mask = dropout_forward(A_curr, dropout_prob)
        # Save dropout mask to memory for backpropagation
        memory["dropout_mask" + str(layer_idx)] = dropout_mask
        
    return A_curr, Z_norm, mean, variance

def full_forward_propagation(X, params_values, nn_architecture, gamma_values, beta_values, dropout_prob=0.0):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        gamma = gamma_values["gamma" + str(layer_idx)] if "gamma" + str(layer_idx) in gamma_values else None
        beta = beta_values["beta" + str(layer_idx)] if "beta" + str(layer_idx) in beta_values else None
        
        A_curr, Z_norm, mean, variance = single_layer_forward_propagation(A_prev, W_curr, b_curr, gamma, beta, activ_function_curr, dropout_prob)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_norm
        memory["mean" + str(layer_idx)] = mean
        memory["variance" + str(layer_idx)] = variance
       
    return A_curr, memory

def get_cost_value(Y_hat, Y, params_values, lambda_reg):
    m = Y_hat.shape[1]
    cross_entropy_cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    
    # L2 regularization term
    l2_regularization = 0
    for layer_idx in range(1, len(params_values) // 2 + 1):
        W_curr = params_values['W' + str(layer_idx)]
        l2_regularization += np.sum(np.square(W_curr))

    cost = cross_entropy_cost + (lambda_reg / (2 * m)) * l2_regularization
    return np.squeeze(cost)
    
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev,  gamma, beta, mean, variance, activation="relu",  lambda_reg=0., dropout_mask=None):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    if dropout_mask is not None:
        dA_curr = dropout_backward(dA_curr, dropout_mask)
        
    dZ_tilde = backward_activation_func(dA_curr, Z_curr)
    dZ, dgamma, dbeta = batch_norm_backward(dZ_tilde, Z_curr, mean, variance, gamma, beta)
    # L2 regularization term gradient
    dW_reg = (lambda_reg / m) * W_curr
    
    dW_curr = np.dot(dZ, A_prev.T) / m + dW_reg
    db_curr = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ)

    return dA_prev, dW_curr, db_curr, dgamma, dbeta

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture,  gamma_values, beta_values):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
   
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_norm = memory["Z" + str(layer_idx_curr)]
        mean = memory["mean" + str(layer_idx_curr)]
        variance = memory["variance" + str(layer_idx_curr)]
        dropout_mask = memory.get("dropout_mask" + str(layer_idx_curr))

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        gamma = gamma_values["gamma" + str(layer_idx_curr)] if "gamma" + str(layer_idx_curr) in gamma_values else None
        beta = beta_values["beta" + str(layer_idx_curr)] if "beta" + str(layer_idx_curr) in beta_values else None
        
        dA_prev, dW_curr, db_curr, dgamma, dbeta = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_norm, A_prev, gamma, beta, mean, variance, activ_function_curr,dropout_mask)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
        if gamma is not None:
            grads_values["dgamma" + str(layer_idx_curr)] = dgamma
            grads_values["dbeta" + str(layer_idx_curr)] = dbeta
    
    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
    return params_values, cost_history, accuracy_history
