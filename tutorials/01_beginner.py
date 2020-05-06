import tensorflow as tf
import numpy
import random

debug_display = True
if debug_display:
    numpy.set_printoptions(linewidth=100000)
    displaynum = random.randrange(0,60000-1)

# import the keras handwritten numbers training data
# size 60,000 training images and 10,000 test images
# 28x28 arrays of 0-255 -> single integer
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if debug_display:
    print("\nThis is a randomly selected piece of training data:")
    print("X = ")
    print(x_train[displaynum])
    print("Y = ")
    print(y_train[displaynum])

# conver the samples from integers to floating point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0

if debug_display:
    print("\nNow adjusted to floating point:")
    print("X = ")
    print(x_train[displaynum])
    print("Y = ")
    print(y_train[displaynum])

# build the model by stacking layers
model = tf.keras.models.Sequential([

  # 1. flatten the input
  # grayscale implies (batch, 1) format, channels last
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  # 2. add a regular, densely connected NN
  # units=128, dimensionality of output space ## why 128??
  # activation by default is linear, here we use rectified linear
  tf.keras.layers.Dense(128, activation='relu'),

  # 3. reduce overfitting by allowing dropout
  # this is an effective way of reducing overfitting (similar to ensembles)
  tf.keras.layers.Dropout(0.2),

  # 4. add a final output layer with one node per desired output (integers 0-9)
  # TODO(zpm): could you change this to 3 and get the binary encoding for the numbers?
  tf.keras.layers.Dense(10)

])

# look at what the model is doing by default (without training)
predictions = model(x_train[0:5])
print("\nThe untrained predictions for the first 5 pieces of training data:")
print(predictions)

# turn the outputs into probabilities
# it's apparently bad practice to bake this into the last layer of the NN for reasons i don't fully understand yet
softmax_probabilities = tf.nn.softmax(predictions)
print("\nProbabilistic output using softmax:")
print(softmax_probabilities)

# set up the loss function for the newtork
# the untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to tf.log(1/10) ~= 2.3
# doing the SparseCategoricalCrossentropy from logits is recommended, for the same (unknown) reason as above
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_fn(y_train[0:5], predictions)
print("\nInitial loss (should be close to 2.3):")
print(initial_loss)

# compile the full model
# TODO(zpm): learn why adam is the best here
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# fit the model to the training data
print("\nTrain the model:")
model.fit(x_train, y_train, epochs=5)

# evaluate the model with test data
print("\nEvaluate the model:")
model.evaluate(x_test, y_test)