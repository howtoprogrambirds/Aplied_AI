import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt


def create_dense(layer_sizes, image_size):
    # It creates a multi-layer network that always has appropriate
    # input and output layers for our MNIST task. Specifically,
    # the models will have:
    # - input vector length of 784
    # - output vector of length 10 that uses a one-hot encoding and the
    #   softmax activation function
    # - a number of layers with the widths specified by the input array
    #   all using the sigmoid activation function.
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))

    for s in layer_sizes[1:]:
        model.add(Dense(units=s, activation='sigmoid'))

    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def evaluate(model, x_train, x_test, y_train, y_test, batch_size=128, epochs=5):
    # prints a summary of the model, trains the model, graphs the
    # training and validation accuracy, and prints a summary of its
    # performance on the test data.
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

# Preparing the dataset
# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train

image_size = 28 * 28 #length from vector of a image
x_train = x_train.reshape(x_train.shape[0], image_size) #flatten the train images
x_test = x_test.reshape(x_test.shape[0], image_size) #flatten the test images

# Convert class vectors to binary class matrices.
# The idea is to make a "one-hot encoded" vector. This is
# a vector with the same length as the number of categories
# we are using, and force the model to set exactly one of the
# positions in the vector to 1 and the rest to 0.

# This is the one-hot version of: [5, 0, 4, 1, 9]
# [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# I have chosen for this because if the algorithm predicts "6" when
# it should be "2" it is wrong to say that the algorithm is "off by 4"
# it simply predicted the wrong category.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

for layers in range(1, 5):
    model = create_dense([32] * layers, image_size)
    evaluate(model, x_train, x_test, y_train, y_test)

model = create_dense([32, 32, 32], image_size)
evaluate(model, x_train, x_test, y_train, y_test, epochs=40)

for nodes in [32, 64, 128, 256, 512, 1024, 2048]:
    model = create_dense([nodes], image_size)
    evaluate(model, x_train, x_test, y_train, y_test)

nodes_per_layer = 32
for layers in [1, 2, 3, 4, 5]:
    model = create_dense([nodes_per_layer] * layers, image_size)
    evaluate(model, x_train, x_test, y_train, y_test, epochs=10*layers)

nodes_per_layer = 128
for layers in [1, 2, 3, 4, 5]:
    model = create_dense([nodes_per_layer] * layers, image_size)
    evaluate(model, x_train, x_test, y_train, y_test, epochs=10*layers)

nodes_per_layer = 512
for layers in [1, 2, 3, 4, 5]:
    model = create_dense([nodes_per_layer] * layers, image_size)
    evaluate(model, x_train, x_test, y_train, y_test, epochs=10 * layers)

model = create_dense([32] * 5, image_size)
evaluate(model, x_train, x_test, y_train, y_test, batch_size=16, epochs=50)

# if you could see the plot with a network of 3 layers, 512 nodes and 30
# epochs is the best