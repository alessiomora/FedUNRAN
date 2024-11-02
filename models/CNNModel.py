import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, only_digits=True, seed=1):
        super().__init__()
        data_format = 'channels_last'
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.HeNormal(
                                                seed=seed),
                                            )

        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.HeNormal(
                                                seed=seed),
                                            )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                                  data_format=data_format)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                                  data_format=data_format)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10 if only_digits else 47)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
