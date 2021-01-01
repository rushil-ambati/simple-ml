import numpy as np
import matplotlib.pyplot as plt

debug = True

iterations = 10000
lr = 0.001

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# inputs
X_train = np.array([[1, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 1, 0],
                            [1, 0, 1, 1],
                            [0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [0, 0, 1, 1],
                            [1, 1, 0, 1],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [1, 0, 0, 1]])
# labels
y_train = np.array([[0],
                             [0],
                             [0],
                             [1],
                             [1],
                             [1],
                             [0],
                             [1],
                             [1],
                             [0],
                             [1],
                             [0],
                             [1],
                             [1],
                             [0]])

np.random.seed(1)

# generate random initial weights with mean 0
weights = 2 * np.random.random((4, 1)) - 1

if debug == True:
    print("Initial weights:")
    print(weights)

x_err = np.arange(0, iterations, 1)
y_err = np.array([])

for _ in range(iterations):
    input_layer = X_train
    outputs = sigmoid(np.dot(input_layer, weights))

    error = y_train - outputs
    
    mse = np.square(outputs - y_train).mean() # root mean squared error
    y_err = np.append(y_err, mse)
    
    adjustments = lr * error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)
if debug == True:
    print("Final weights:")
    print(weights)

if debug == True:
    print()
    print("Training outputs:")
    print(outputs)
    print("Training error:")
    print(error)

X_validate = np.array([[1, 0, 0, 0],
                       [1, 0, 1, 1]])
y_validate = np.array([[0],
                       [1]])

input_layer = X_validate
outputs = sigmoid(np.dot(input_layer, weights))

mse = np.square(outputs - y_validate).mean() # mean squared error

if debug == True:
    print()
    print("Testing outputs:")
    print(outputs)
    print("Testing error:")
    print(mse)
  
plt.plot(x_err, y_err, 'r-')
plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
plt.title("Backpropagation Performance (Error)")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.xlim(0)
plt.ylim(0)

plt.savefig('error_plot.png')
plt.show()