import tensorflow as tf


class ExciteBlock(tf.keras.layers.Layer):
    """
    A layer that learns to perform a series of transformations on a vector input
    (typically a vector containing the averages of a number of feature maps) in
    order to extract useful signals to scale/weight the original feature maps.
    """

    def __init__(self, config, **kwargs):
        super(ExciteBlock, self).__init__(**kwargs)
        self.config = config
        self.ratio = config.compression_ratio

    def build(self, input_shape):
        compress_dim = input_shape[-1] // self.ratio
        uncompress_dim = input_shape[-1]

        ew1_init = tf.keras.initializers.he_normal()
        eb1_init = tf.keras.initializers.Zeros()
        self.ew1 = self.add_weight(
            shape=(compress_dim, input_shape[-1]),
            initializer=ew1_init,
            trainable=True,
            name='excite_w1',
            regularizer=tf.keras.regularizers.l2(self.config.l2_strength)
        )
        self.eb1 = self.add_weight(
            shape=(compress_dim,),
            initializer=eb1_init,
            trainable=True,
            name='excite_b1',
            regularizer=tf.keras.regularizers.l2(self.config.l2_strength)
        )
        ew2_init = tf.keras.initializers.he_normal()
        eb2_init = tf.keras.initializers.Zeros()
        self.ew2 = self.add_weight(
            shape=(uncompress_dim, compress_dim),
            initializer=ew2_init,
            trainable=True,
            name='excite_w2',
            regularizer=tf.keras.regularizers.l2(self.config.l2_strength)
        )
        self.eb2 = self.add_weight(
            shape=(uncompress_dim,),
            initializer=eb2_init,
            trainable=True,
            name='excite_b2',
            regularizer=tf.keras.regularizers.l2(self.config.l2_strength)
        )

    def call(self, inputs, training=False):
        x = tf.linalg.matvec(self.ew1, inputs) + self.eb1
        x = tf.nn.relu(x)

        if training:
            x = tf.nn.dropout(x, rate=self.config.dropout_rate)

        x = tf.linalg.matvec(self.ew2, x) + self.eb2
        x = tf.math.sigmoid(x + 1e-08)
        
        return x


def mask_embedding(embedding_layer, allowed_indices, mask):

    for index in allowed_indices:
        mask = tf.tensor_scatter_nd_update([[index]], mask, [1.0])

    masked_embeddings = tf.multiply(embedding_layer.embeddings, mask[:, tf.newaxis])
    return masked_embeddings


class SqueezeExciteCNN(tf.keras.Model):
    """Simple Squeeze and Excitation 1D CNN for text classification"""

    def __init__(self, config, concept2index, #valid_ids,
                 **kwargs):
        super(SqueezeExciteCNN, self).__init__(**kwargs)
        self.config = config
        self.concept2index = concept2index
        self.node_embeddings = tf.keras.layers.Embedding(
            len(self.concept2index), self.config.concept_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(0, 0.01),
            mask_zero=True,
            trainable=True
        )
        self.conv1d = tf.keras.layers.Conv1D(
            self.config.num_filters,
            1,
            activation=tf.keras.activations.elu,
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_strength),
            bias_regularizer=tf.keras.regularizers.l2(config.l2_strength),
            name='CNN_conv1d'
        )
        self.excite = ExciteBlock(config)
        self.dense1 = tf.keras.layers.Dense(
            self.config.concept_dim,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.1
            ),
            bias_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.01
            ),
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_strength),
            bias_regularizer=tf.keras.regularizers.l2(config.l2_strength),
            name='CNN_dense1'
        )
        # ensure node embeddings exist for first pass
        _ = self.node_embeddings(
            tf.range(self.node_embeddings.input_dim)
        )

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        non_zero = tf.cast(tf.not_equal(x, tf.zeros_like(x)), tf.float32)
        sum = tf.math.reduce_sum(x, axis=-2)
        num = tf.math.reduce_sum(non_zero, axis=-2)
        mu = tf.math.divide_no_nan(sum, num)
        x2 = self.excite(mu, training=training)
        x = tf.reduce_max(x, axis=-2)
        x = tf.multiply(x, x2)

        if training:
            x = tf.nn.dropout(x, rate=self.config.dropout_rate)

        x = self.dense1(x)

        if training:
            x = tf.nn.dropout(x, rate=self.config.dropout_rate)

        x = tf.nn.l2_normalize(x, axis=-1)

        if not training:
            x = tf.matmul(x, self.node_embeddings.embeddings, transpose_b=True)

        return x

    def get_config(self):
        return {
            'num_filters': self.num_filters,
            'concept_dim': self.concept_dim,
            'squeeze_excite_ratio': self.ratio
        }

    def save(self):
        self.save_weights(f'{self.config.param_dir}/{self.config.model_name}/cnn_model.h5')

    def load(self):
        self.load_weights(f'{self.config.param_dir}/{self.config.model_name}/cnn_model.h5')
        