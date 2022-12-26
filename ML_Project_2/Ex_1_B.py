import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
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


print ('Train size:', x_train.shape[0])
print ('Test size:', x_test.shape[0])
batch_size = 128
epochs = 5
Layers = 40

for activation in ['relu', 'tanh', 'sigmoid']:
    model = Sequential()
    model.add(Dense(32, activation=activation, input_shape=(784,)))
    for i in range(Layers-2):
        model.add(Dense(32,activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    plt.plot(history.history['val_acc'])
    score = model.evaluate(x_test, y_test, verbose=100)
    print()
    print('Test loss:', round(score[0], 3))
    print('Test accuracy:', round(score[1], 3))

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['relu', 'tanh', 'sigmoid'], loc='upper left')
fig1=plt.gcf()
plt.show()
fig1.savefig('Fig_1_B_'+str(Layers)+'_Layers')
# plot_model(model, to_file='model_plot_5_Layers.png', show_shapes=True, show_layer_names=True)