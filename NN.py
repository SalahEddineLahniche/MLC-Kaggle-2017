import numpy as np
import tensorflow as tf
import pandas as pd
from math import floor


## loading the data and preparing it for the learning
df = pd.read_csv('data/train.csv').astype('float32')

X = df.drop('power_increase', axis=1)
y = df['power_increase']

train_size = 0.85
train_cnt = floor(X.shape[0] * train_size)

X_train = X.iloc[0:train_cnt].values
y_train = y.iloc[0:train_cnt].values

X_test = X.iloc[train_cnt:].values
y_test = y.iloc[train_cnt:].values

X = X.values
y = y.values
## constucting the neural network
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

n_hidden_1 = 500
n_input = X.shape[1]
n_classes = 1

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

training_epochs = 5000
display_step = 500
batch_size = 256


x = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# print(x)
# print(y)

predictions = multilayer_perceptron(x, weights, biases)
cost = tf.square(Y - predictions, name="cost")
cost = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(X_train) / batch_size)
        x_batches = np.array_split(X_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], 
                            feed_dict={
                                x: batch_x, 
                                Y: batch_y,
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, Y: y_test}))