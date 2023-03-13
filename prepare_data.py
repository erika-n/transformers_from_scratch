import numpy as np

def getXY():
    input_length = 128


    encoded_tokens = np.load('encoded_tokens.npy')

    inputs = encoded_tokens.shape[0]//input_length

    X = np.reshape(encoded_tokens[:inputs*input_length], (-1, input_length))
    encoded_tokens = np.roll(encoded_tokens, -1)
    Y = np.reshape(encoded_tokens[:inputs*input_length], (-1, input_length))
    return X, Y

def getTestTrain(testpct):
    X, Y = getXY()

    p = np.random.permutation(X.shape[0])
    X = X[p]
    Y = Y[p]


    n_test = int(testpct*X.shape[0])
    X_test = X[:n_test]
    Y_test = Y[:n_test]
    X_train = X[n_test:]
    Y_train = Y[n_test:]
    return X_test, Y_test, X_train, Y_train

if __name__ == '__main__':
    Xt, Yt, X, Y = getTestTrain(0.02)
    print("Xt, Yt, X, Y")
    print(Xt.shape, Yt.shape, X.shape, Y.shape)