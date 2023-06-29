import tensorflow as tf
import random
import numpy as np


class AdjacencyAggregator(tf.keras.layers.Layer):
    """
    Layer that aggregates multiple adjacency matrices into a single one using
    soft attention.
    """

    def __init__(self, config, **kwargs):
        super(AdjacencyAggregator, self).__init__(**kwargs)
        self.config = config
        # self.temperature_scaling = TemperatureScaling()

    def build(self, input_shape):
        # build custom 1x1 convolutional layer to reduce the number of adjacency
        # matrices into a single one. The kernel is initialized with random
        # values and then normalized to sum to 1
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, 1, input_shape[0][-3], self.config.max_path_len),
            # as in the paper: https://doi.org/10.1016/j.neunet.2022.05.026
            initializer=tf.keras.initializers.RandomNormal(0, 0.1),
            trainable=True
        )

    def call(self, inputs, training=False):
        # reshape inputs from
        # (batch_size, num_adj_matrices, num_nodes, num_nodes)
        # to
        # (batch_size, num_nodes, num_nodes, num_adj_matrices)
        x = tf.transpose(inputs[0], perm=[0, 2, 3, 1])
        soft = tf.nn.softmax(tf.nn.relu(self.kernel), axis=-2)
        # 1x1 convolution using the custom kernel to softly select the adjacency 
        # matrices
        conv = tf.nn.conv2d(x, soft, data_format='NHWC', strides=1,
                            padding='VALID')
        # transpose back to (batch_size, num_adj_matrices, num_nodes, num_nodes)
        x = tf.transpose(conv, perm=[0, 3, 1, 2])
        # Compute the matrix product for each batch
        batch_size, num_adj_matrices, num_nodes, _ = x.shape
        result = tf.eye(num_nodes, batch_shape=[batch_size], dtype=x.dtype)
        for i in range(num_adj_matrices):
            result = tf.linalg.matmul(result, x[:, i, :, :])
        meta_adjacency = result
        # row-wise degree normalization of the adjacency matrix.
        normed_adjacency = tf.matmul(tf.linalg.inv(inputs[1]), meta_adjacency)

        return normed_adjacency

    def get_config(self):
        config = super(AdjacencyAggregator, self).get_config()
        config.update({
            'filters': self.filters,
            'sd': self.sd
        })
        return config


class MetaPathHead(tf.keras.Model):
    """
    A head that uses adjacency matrices to aggregate information from multiple
    adjacency matrices into meta-paths.
    """
    def __init__(self, config, **kwargs):
        super(MetaPathHead, self).__init__(**kwargs)
        self.config = config
        self.adjacency_aggregator = AdjacencyAggregator(config)
        self.input_dropout = InputDropout(config.input_dropout_rate)

    def build(self, input_shape):
        head_id = '%0.9d' % random.randint(0, 999999999)
        KQ_dim = input_shape[2][-1]//self.config.num_heads
        init = tf.keras.initializers.he_uniform()
        bias_init = tf.keras.initializers.Zeros()
        self.Wv = tf.Variable(
            initial_value=init(shape=(input_shape[2][-1], KQ_dim),
                               dtype=tf.float32),
            trainable=True,
            name=f"Wv_{head_id}"
        )
        self.Bv = tf.Variable(
            initial_value=bias_init(shape=(input_shape[2][-2], KQ_dim),
                                    dtype=tf.float32),
            trainable=True,
            name=f"Bv_{head_id}"
        )

    def call(self, inputs, training=False):
        adjacency_matrix = self.adjacency_aggregator([inputs[0], inputs[1]])
        x = self.input_dropout(inputs[2], training=training)
        V = tf.matmul(x, self.Wv) + self.Bv

        if training:
            V = tf.nn.dropout(V, rate=self.config.dropout_rate)

        return tf.matmul(adjacency_matrix, V)

    def get_config(self):
        config = super(MetaPathHead, self).get_config()
        config.update({'max_path_len': self.max_path_len,
                       'num_heads': self.h})
        return config


class RowWiseFFN(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(RowWiseFFN, self).__init__(**kwargs)
        self.config = config
        self.norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True)

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(
            input_shape[-1]*2,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            activation=tf.nn.gelu,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_strength) \
                if self.config.l2_strength is not None else None
        )
        self.dense2 = tf.keras.layers.Dense(
            input_shape[-1],
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_strength) \
                if self.config.l2_strength is not None else None
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = tf.nn.dropout(x, rate=self.config.dropout_rate)
        x = self.norm(x)
        x = self.dense2(x)
        return x

class InputDropout(tf.keras.layers.Layer):
    
    def __init__(self, dropout_rate, **kwargs):
        super(InputDropout, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=False):
        if not training:
            return inputs
        # Create a dropout mask for the entire sequence except for the first
        # vector
        dropout_mask = tf.random.uniform(
            shape=(inputs.shape[0], inputs.shape[1] - 1)
        ) > self.dropout_rate
        # Add a True value at the beginning of the dropout_mask for each
        # sequence in the batch
        first_vector_mask = tf.ones(shape=(inputs.shape[0], 1), dtype=tf.bool)
        dropout_mask = tf.concat([first_vector_mask, dropout_mask], axis=-1)
        # Expand the dimensions of the dropout_mask to match the input shape
        dropout_mask = tf.expand_dims(dropout_mask, axis=-1)
        dropout_mask = tf.cast(dropout_mask, inputs.dtype)
        # Apply the dropout_mask to the inputs
        outputs = inputs * dropout_mask * (1.0 + self.dropout_rate)
        return outputs

    def set_new_dropout_rate(self, dropout_rate):
        if dropout_rate < 0.1:
            self.dropout_rate = 0.1
        else:
            self.dropout_rate = dropout_rate


class MetaPathTransformer(tf.keras.Model):
    """
    A transformer that learns meta-paths and learns to extract
    meta-path-specific signals for task-specific message passing between nodes.
    """
    def __init__(self, config, **kwargs):
        super(MetaPathTransformer, self).__init__(**kwargs)
        self.config = config
        # self.input_dropout = InputDropout(self.config.input_dropout_rate)
        self.heads = [MetaPathHead(config) for _ in range(config.num_heads)]
        # self.W0 = tf.keras.layers.Dense(config.concept_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True)
        self.norm2 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, trainable=True)
        self.ffn = RowWiseFFN(config)

    def build(self, input_shape):
        init = tf.keras.initializers.he_uniform()
        self.W0 = tf.Variable(
            initial_value=init(shape=(input_shape[2][-1], input_shape[2][-1]),
                               dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs, training=False):
        """Pre-norm"""
        adjacency = inputs[0]
        degree = inputs[1]
        x = input = inputs[2]
        # randomly drop entire heads during training.
        self_attention = tf.concat(
            [
                (head_output * (1 + self.config.head_dropout_rate)) * (
                        np.random.rand() >= self.config.head_dropout_rate)
                if training else head_output
                for head in self.heads
                for head_output in [head([adjacency, degree, x],
                                         training=training)]
            ],
            axis=-1)
        x = tf.matmul(self_attention, self.W0)
        residual = tf.add(x, input)
        x = self.norm2(residual)
        x = self.ffn(x, training=training)
        x = tf.add(x, residual)
        return x

    def get_config(self):
        config = super(MetaPathTransformer, self).get_config()
        return config

    def save(self):
        self.save_weights(f'{self.config.param_dir}/{self.config.model_name}/meta_path_transformer.h5')

    def load(self):
        self.load_weights(f'{self.config.param_dir}/{self.config.model_name}/meta_path_transformer.h5')

    def plot_kernel_weights_lattice(self, name=None):
        kernels = [tf.nn.softmax(tf.nn.relu(head.adjacency_aggregator.kernel),
                                 axis=-2) for head in self.heads]

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        # get subplot dims
        n = len(kernels)
        labels = [
            'identity',
            'gda2gene',
            'gda2hpo',
            'gene2protein',
            'subclassof',
            'gene2gda',
            'hpo2gda',
            'protein2gene',
            'superclassof'
        ]

        # Find the global min and max values across all kernels
        global_min = min(np.min(kernel) for kernel in kernels)
        global_max = max(np.max(kernel) for kernel in kernels)

        # plot lattice of all the kernels
        fig, axs = plt.subplots(1, n, figsize=(self.config.max_path_len * n,
                                               len(labels) * 1.25))
        fig.subplots_adjust(wspace=0.5)
        for i, kernel in enumerate(kernels):
            kernel = np.squeeze(kernel)
            sns.heatmap(kernel, annot=True, fmt='.2f', cmap='Blues', ax=axs[i],
                        vmin=global_min, vmax=global_max)
            # label the y axis with the respective relationships, from top to
            # bottom, only for the first subplot
            if i == 0:
                axs[i].set_yticks(np.arange(len(labels)) + 0.5)
                axs[i].set_yticklabels(labels)
            else:
                axs[i].set_yticklabels([])
            axs[i].set_title(f'MetaPathHead {i}')
        filepath = f'{self.config.figures_dir}/{self.config.model_name}/kernels.png' if name is None \
            else f'{self.config.figures_dir}/{self.config.model_name}/kernels_{name}.png'
        plt.savefig(filepath)

