import tensorflow as tf
import flwr as fl

model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    weights=None,
    classes=10
)

model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



# parameters are the weights that server sends to the client
# Config are the dictionary of instruction like number of epoch, learning rate, etc hyperparameters

class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())