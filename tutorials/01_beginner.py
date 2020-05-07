import tensorflow as tf
import numpy
import random

debug_display = True
if debug_display:
    numpy.set_printoptions(linewidth=100000)
    displaynum = random.randrange(0,60000-1)

# import the keras handwritten numbers training data
# 60,000 training images and 10,000 test images
# 28x28 arrays of 0-255 -> single integer
mnist = tf.keras.datasets.mnist
(x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()

if debug_display:
    print("\nThis is a randomly selected piece of training data:")
    print("X = ")
    print(x_train_raw[displaynum])
    print("Y = ")
    print(y_train[displaynum])

# convert the samples from integers to floating point numbers
x_train, x_test = x_train_raw / 255.0, x_test_raw / 255.0

# if debug_display:
#     print("\nNow adjusted to floating point:")
#     print("X = ")
#     print(x_train[displaynum])
#     print("Y = ")
#     print(y_train[displaynum])

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
print("\nProbabilistic output using softmax (%):")
with numpy.printoptions(formatter={'float': '{: 0.8f}'.format}):
    print(softmax_probabilities.numpy() * 100)

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
# for this example, epochs=1 is 90%, 5 is 97%, 10 is 98%
print("\nTrain the model:")
train_result = model.fit(x_train, y_train, epochs=5)

# evaluate the model with test data
print("\nEvaluate the model:")
eval_result = model.evaluate(x_test, y_test)

# print a few examples of losses
print("\nA few examples of misclassified items from training set:")
num_printed = 0
for i in range(0, len(x_train)):

    eval_result = model.evaluate(x_train[i:i+1], y_train[i:i+1], verbose=0)
    if eval_result[1] < 1.0: # accuracy fail

        # turn the outputs into probabilities
        final_predictions = model(x_train[i:i+1])
        final_softmax_probabilities = tf.nn.softmax(final_predictions)
        print("\nProbabilistic output using softmax for x_train[" + str(i) + "]")
        print("y_train=" + str(y_train[i]))
        with numpy.printoptions(formatter={'float': '{: 0.8f}'.format}):
            print(final_softmax_probabilities.numpy() * 100)

        print(x_train_raw[i])

        num_printed += 1
        if num_printed == 5:
            break

# print a few examples of losses
print("\nA few examples of misclassified items from test set:")
num_printed = 0
for i in range(0, len(x_test)):

    eval_result = model.evaluate(x_test[i:i+1], y_test[i:i+1], verbose=0)
    if eval_result[1] < 1.0: # accuracy fail

        # turn the outputs into probabilities
        final_predictions = model(x_test[i:i+1])
        final_softmax_probabilities = tf.nn.softmax(final_predictions)
        print("\nProbabilistic output using softmax for x_test[" + str(i) + "]")
        print("y_test=" + str(y_test[i]))
        with numpy.printoptions(formatter={'float': '{: 0.8f}'.format}):
            print(final_softmax_probabilities.numpy() * 100)

        print(x_test_raw[i])

        num_printed += 1
        if num_printed == 5:
            break