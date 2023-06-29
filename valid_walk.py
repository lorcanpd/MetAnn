import tensorflow as tf
import numpy as np
import random

from gnn import RowWiseFFN

"""output classification layer for GNN. 1 represents a valid walk, 0 represents 
an invalid walk. Takes in an n by m matric and outputs a single value for binary 
classification."""


class AttentionHead(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        head_id = '%0.9d' % random.randint(0, 999999999)
        KQ_dim = input_shape[-1] // self.config.num_heads
        init = tf.keras.initializers.he_uniform()
        bias_init = tf.keras.initializers.Zeros()
        self.Wq = tf.Variable(
            initial_value=init(shape=(input_shape[-1], KQ_dim),
                                 dtype=tf.float32),
            trainable=True,
            name=f"Wq_{head_id}"
        )
        self.Bq = tf.Variable(
            initial_value=bias_init(shape=(input_shape[-2], KQ_dim),
                                    dtype=tf.float32),
            trainable=True,
            name=f"Bq_{head_id}"
        )
        self.Wk = tf.Variable(
            initial_value=init(shape=(input_shape[-1], KQ_dim),
                                 dtype=tf.float32),
            trainable=True,
            name=f"Wk_{head_id}"
        )
        self.Bk = tf.Variable(
            initial_value=bias_init(shape=(input_shape[-2], KQ_dim),
                                    dtype=tf.float32),
            trainable=True,
            name=f"Bk_{head_id}"
        )
        self.Wv = tf.Variable(
            initial_value=init(shape=(input_shape[-1], KQ_dim),
                               dtype=tf.float32),
            trainable=True,
            name=f"Wv_{head_id}"
        )
        self.Bv = tf.Variable(
            initial_value=bias_init(shape=(input_shape[-2], KQ_dim),
                                    dtype=tf.float32),
            trainable=True,
            name=f"Bv_{head_id}"
        )

    def call(self, inputs, training=False):
        Q = tf.matmul(inputs, self.Wq) + self.Bq
        K = tf.matmul(inputs, self.Wk) + self.Bk
        V = tf.matmul(inputs, self.Wv) + self.Bv

        if training:
            Q = tf.nn.dropout(Q, rate=self.config.dropout_rate)
            K = tf.nn.dropout(K, rate=self.config.dropout_rate)
            V = tf.nn.dropout(V, rate=self.config.dropout_rate)

        if self.config.l2_strength:
            self.add_loss(tf.nn.l2_loss(self.Wq) * self.config.l2_strength)
            self.add_loss(tf.nn.l2_loss(self.Wk) * self.config.l2_strength)
            self.add_loss(tf.nn.l2_loss(self.Wv) * self.config.l2_strength)
            self.add_loss(tf.nn.l2_loss(self.Bq) * self.config.l2_strength)
            self.add_loss(tf.nn.l2_loss(self.Bk) * self.config.l2_strength)
            self.add_loss(tf.nn.l2_loss(self.Bv) * self.config.l2_strength)

        attention = tf.matmul(Q, K, transpose_b=True)
        if training:
            attention = tf.nn.dropout(attention, rate=self.config.dropout_rate)

        attention = attention / tf.math.sqrt(tf.cast(K.shape[-1], tf.float32))
        attention = tf.nn.softmax(attention)
        return tf.matmul(attention, V)

class Decoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.config = config
        self.norm1 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True)
        self.norm2 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True)
        self.ffn = RowWiseFFN(config)

    def build(self, input_shape):
        self.heads = [AttentionHead(self.config) for _
                      in range(self.config.num_heads)]
        self.W0 = tf.Variable(
            initial_value=tf.keras.initializers.he_uniform()(
                shape=(input_shape[-1], input_shape[-1]),
                dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs, training=False):
        # Sublayer 1: Multi-Head Attention
        self_attention = tf.concat(
            [
                (head_output * (1 + self.config.head_dropout_rate)) * (
                            np.random.rand() >= self.config.head_dropout_rate)
                if training else head_output
                for head in self.heads
                for head_output in [head(inputs, training=training)]
            ],
            axis=-1
        )

        head_output_transformed = tf.matmul(self_attention, self.W0)
        residual1 = inputs + head_output_transformed  # Residual Connection
        # Apply LayerNorm after the first sub-layer (Self-Attention)
        x = self.norm1(residual1)
        # Sublayer 2: Position-Wise Feed-Forward Networks
        ffn_output = self.ffn(x, training=training)
        residual2 = x + ffn_output  # Residual Connection
        # Apply LayerNorm after the second sub-layer (FFN)
        output = self.norm2(residual2)

        return output

class OutputLayer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.config = config
        self.norm1 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True
        )

    def build(self, input_shape):
        self.decoder = Decoder(self.config)
        self.dense1 = tf.keras.layers.Dense(
            input_shape[-1]*2,
            activation=tf.nn.gelu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='output_dense1',
            trainable=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_strength) \
                if self.config.l2_strength is not None else None
        )
        self.dense2 = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='output_dense2',
            trainable=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_strength) \
                if self.config.l2_strength is not None else None
        )

    def call(self, inputs, training=False):
        x = inputs
        x = self.decoder(x, training=training)
        # get the first vector of each sequence in the batch
        x = tf.gather(x, 0, axis=1)
        x = self.dense1(x)

        if training:
            x = tf.nn.dropout(x, rate=self.config.dropout_rate)

        x = self.norm1(x)
        x = self.dense2(x)
        return tf.squeeze(x)

    def get_config(self):
        return self.config

    def save(self):
        self.save_weights(f'{self.config.param_dir}/{self.config.model_name}/valid_walk_decoder.h5')

    def load(self):
        self.load_weights(f'{self.config.param_dir}/{self.config.model_name}/valid_walk_decoder.h5')

