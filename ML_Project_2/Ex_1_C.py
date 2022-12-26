import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


# function to obtain grads for each parameter
def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    return np.array(output_grad)


import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Train size:', x_train.shape[0])
print('Test size:', x_test.shape[0])
batch_size = 128
epochs = 3
Layer_Depth = 5

for activation in ['relu', 'tanh', 'sigmoid']:
    max_gradient_layer_i = np.zeros(Layer_Depth)
    model = Sequential()
    model.add(Dense(32, activation=activation, input_shape=(784,)))
    for i in range(Layer_Depth - 2):
        model.add(Dense(32, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        # batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    grads = get_gradients(model, x_train, y_train)
    j=0
    for i in range(1, len(grads), 2):

        # print ("************"+str(j))
        max_gradient_layer_i[j] = np.max(grads[i])
        j=j+1
    plt.plot(max_gradient_layer_i)

plt.title('Max Values of Gradients,'+str(Layer_Depth)+'_Layers')
plt.ylabel('Max Gradient')
plt.xlabel('Layer')
plt.legend(['relu', 'tanh', 'sigmoid'], loc='upper left')
fig1=plt.gcf()
plt.show()
fig1.savefig('Fig_1_C_'+str(Layer_Depth)+'_Layers')
