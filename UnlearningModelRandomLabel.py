import tensorflow as tf

class UnlearningModelRandomLabel(tf.keras.Model):
    """
    Versione che usa solo la KL divergence sui forgetting data
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        # self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kl_div = tf.keras.losses.KLDivergence()


    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # x, y = data   # x [batch_size, 28, 28, 1]  y [batch_size] oppure [batch_size, 1]
        x, y, z = data # dove z [batch_size]
        forgetting = z

        with tf.GradientTape() as tape:
            output_unlearning_model_f = self.model(x, training=True)  # Forward pass
            output_unlearning_model = tf.nn.softmax(output_unlearning_model_f)
            # tf.print("output_unlearning_model", output_unlearning_model, summarize=-1)

            batch_size = tf.shape(output_unlearning_model)[0]
            n_classes = tf.shape(output_unlearning_model)[1]

            y = tf.cast(y, tf.int64)
            #forgetting = tf.cast(forgetting, tf.float32)

            random_labels = tf.random.uniform(
                (batch_size, 1),
                minval=0,
                maxval=n_classes,
                dtype=tf.int32)
            index = tf.range(0, batch_size)
            # print(tf.expand_dims(index, 1))
            # print(random_labels)
            indexes = tf.concat([tf.expand_dims(index, 1), random_labels], axis=1)
            # print(indexes)
            output_random = tf.zeros([batch_size, n_classes])
            ones = tf.ones([batch_size])
            output_random = tf.tensor_scatter_nd_update(output_random, indexes, ones)

            # Compute the loss value
            # loss_fn = (1-forgetting)*self.cross_entropy(y_real_label, output_unlearning_model)+ forgetting*self.kl_div(output_random, output_unlearning_model)
            loss = self.kl_div(output_random, output_unlearning_model)
            # tf.print("\n loss ", loss)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        return {"loss": loss}

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        print(data)
        x, y = data

        y_pred = self.model(x, training=False)  # Forward pass
        # self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Return the weights of the local model."""
        return self.model.set_weights(weights)