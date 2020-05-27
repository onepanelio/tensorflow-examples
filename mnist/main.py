import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import argparse

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, help="number of epochs to train model for")
    parser.add_argument("--save_model", default=True, help="save model")
    parser.add_argument("--batch_size", default=64)
    args = parser.parse_args()

    #define and compile model
    model = get_compiled_model()

    #get dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

   

    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val))

    print('\nhistory dict:', history.history)
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    metrics = [
        {'name': 'accuracy', 'value': results[1]},
        {'name': 'loss', 'value': results[0]},
    ]
    with open('/tmp/sys-metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    if args.save_model:
        model.save("/mnt/output/")
