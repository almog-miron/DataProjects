import numpy as np


def backprop(weights, X, y):
    """
    This function receives a set of weights, a matrix with images
    and the corresponding labels. The output should be an array
    with the gradients of the loss with respect to the weights, averaged over
    the samples. It should also output the average loss of the samples.
    :param weights: an array of length L where the n-th cell contains the
    connectivity matrix between layer n-1 and layer n.
    :param X: samples matrix (match the dimensions to your input)
    :param y: corresponding labels
    :return:
    """
    s, h, weights = forward_pass(weights, X)
    weights = np.array(weights)

    hot_y = np.zeros((10, len(y)))
    for i in range(len(y)):
        hot_y[y[i], i] = 1

    loss = np.sum((s[-1] - hot_y) ** 2) / y.shape[0]

    grads = backward_pass(s, h, hot_y, weights, X)
    return grads, loss


def backward_pass(s, h, y, weights, x):
    temp_grads = []
    grads = []
    tot_samples = x.shape[0]
    for l in range(s.shape[0]-1, -1, -1):
        s_layer = s[l]
        h_layer = h[l]
        if l == s.shape[0] - 1:
            gradient = (s_layer - y) * tanh_derivative(h_layer)
        else:
            w_layer = weights[l+1]
            last_grad = temp_grads[-1]
            gradient = np.dot(w_layer.T, last_grad) * tanh_derivative(h_layer)
        temp_grads.append(gradient)

    for l in range(s.shape[0]-1, -1, -1):
        s_layer = s[l]
        if l == 0:
            grads.append(np.dot(temp_grads[0], s_layer.T) / tot_samples)
        else:
            grads.append(np.dot(temp_grads[l], x) / tot_samples)

    return grads


def tanh_derivative(h):
    return 1 - (np.tanh(h) ** 2)


def test(weights, Xtest, ytest):
    """
    This function receives the Network weights, a matrix of samples and
    the corresponding labels, and outputs the classification
    accuracy and mean loss.
    The accuracy is equal to the ratio of correctly labeled images.
    The loss is given the square distance of the last layer activation
    and the 0-1 representation of the true label
    Note that ytest in the MNIST data is given as a vector of labels from 0-9. To calculate the loss you
    need to convert it to 0-1 (one-hot) representation with 1 at the position
    corresponding to the label and 0 everywhere else (label "2" maps to
    (0,0,1,0,0,0,0,0,0,0) etc.)
    :param weights: array of the network weights
    :param Xtest: samples matrix (match the dimensions to your input)
    :param ytest: corresponding labels
    :return:
    """

    # use the function predict to get the predicted label and last layer activation
    yhat, output_activation = predict(weights, Xtest)

    # gets sqr loss
    hot_y = np.zeros((10, len(ytest)))
    for i in range(len(ytest)):
        hot_y[ytest[i], i] = 1

    loss = np.sum((output_activation[-1] - hot_y) ** 2) / ytest.shape[0]

    accuracy = np.sum(yhat == ytest) / ytest.shape[0]

    return accuracy, loss


def predict(weights, X):
    """
    The function takes as input an array of the weights and a matrix (X)
    with images. The outputs should be a vector of the predicted
    labels for each image, and a matrix whose columns are the activation of
    the last layer for each image.
    last_layer_activation should be of size [10 X num_samples]
    predicted_labels should be of size [1 X num_samples] or [10 X num_samples]
    The predicted label should correspond to the index with maximal
    activation in the last layer
    :param weights: array of the network weights
    :param X: samples matrix (match the dimensions to your input)
    :return:
    """
    last_layer_activation, h, weights = forward_pass(weights, X)
    predicted_labels = np.argmax(last_layer_activation[-1].T, 1)

    return predicted_labels, last_layer_activation


def forward_pass(weights, X):
    h = []
    s = []
    s_temp = X.T
    for w in weights:
        h_temp = np.dot(w, s_temp)
        h.append(h_temp)
        s_temp = np.tanh(h_temp)
        s.append(s_temp)

    s = np.array(s, dtype=object)
    h = np.array(h, dtype=object)

    return s, h, weights
