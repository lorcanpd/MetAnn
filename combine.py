import tensorflow as tf

from gnn import MetaPathTransformer
from adjacency import Adjacency
from valid_walk import OutputLayer
from ner_model import SqueezeExciteCNN
from stream_text import FastText

# TODO make this less clumsy... will have to wait unitl after thesis submission
class ForTraining(tf.keras.Model):

    def __init__(self, config, relation_dicts, concept2index, #valid_ids,
                 **kwargs):
        super(ForTraining, self).__init__(**kwargs)
        self.config = config
        self.rel_dict = relation_dicts
        if config.with_random_walks:
            try:
                concept2index['[CLS]']
            except KeyError:
                concept2index['[CLS]'] = len(concept2index)

        self.concept2index = concept2index

        # self.ffn = FFN4BERT(config)
        self.cnn = SqueezeExciteCNN(config, concept2index)#, valid_ids)
        print('CNN initialised')
        self.adjacency = Adjacency(relation_dicts, concept2index, config)
        print('Adjacency matrices created')
        self.gnn = MetaPathTransformer(config)

        if config.with_random_walks:
            print('GNN initialised')
            self.output_layer = OutputLayer(config)

    def call(self, inputs, training=False):
        input_nodes = inputs[0]
        cnn_out = self.cnn(inputs[1], training=training)
        node_samples = self.adjacency.batch_nodes_within_n_hops(
            input_nodes, self.config.max_path_len, training=training
        )
        padded_node_samples = self.adjacency.pad_node_list_batch(node_samples)
        node_embeddings_1 = self.cnn.node_embeddings(padded_node_samples)
        adj_mat_1, degree_1 = self.adjacency.batch_adjacency_matrices(
            node_samples
        )

        gnn_out = self.gnn([adj_mat_1, degree_1, node_embeddings_1], training=training)

        if self.config.with_random_walks:
            walk_labels, walks, adj_mat_2, degree_2 = self.adjacency.get_batch_random_walks(
                self.config.batch_size
            )
            node_embeddings_2 = self.cnn.node_embeddings(walks)
            global_walks = self.gnn([adj_mat_2, degree_2, node_embeddings_2],
                                    training=training)
            walk_predicted = self.output_layer(global_walks, training=training)
            return cnn_out, gnn_out, walk_labels, walk_predicted

        return cnn_out, gnn_out

    def save(self):
        self.cnn.save()
        if self.config.with_random_walks:
            self.gnn.save()
            self.output_layer.save()

    def load(self):
        self.cnn.load()
        if self.config.with_random_walks:
            self.gnn.load()
            self.output_layer.load()

    def replace_embeddings(self):
        """Replace the node embeddings with the aggregated embeddings outputted
        by the GNN"""
        num_embeddings = self.cnn.node_embeddings.embeddings.shape[0]
        new_embeddings_list = []

        print('Replacing embeddings with aggregated embeddings from GNN output')

        for i in range(0, num_embeddings, self.config.batch_size):
            batch_indices = [
                x for x in
                range(i, min(i + self.config.batch_size, num_embeddings))
            ]
            neighbourhood = self.adjacency.batch_nodes_within_n_hops(
                batch_indices, self.config.max_path_len, training=False
            )
            padded_neighbourhood = self.adjacency.pad_node_list_batch(
                neighbourhood
            )
            adj_mat, degree = self.adjacency.batch_adjacency_matrices(
                neighbourhood
            )
            node_embeddings = self.cnn.node_embeddings(padded_neighbourhood)
            new_embeddings_list.extend(
                self.gnn([adj_mat, degree, node_embeddings])[:, 0, :].numpy()
            )
        new_embeddings = tf.convert_to_tensor(
            new_embeddings_list, dtype=self.cnn.node_embeddings.embeddings.dtype
        )
        self.cnn.local_node_embeddings = self.cnn.node_embeddings
        self.cnn.node_embeddings.set_weights([new_embeddings])
        self.save()
        
        
class ForInference(tf.keras.Model):
    """For inference only the CNN and the fasttext model are needed"""

    def __init__(self, config, concept2index,# valid_ids,
                 **kwargs):
        super(ForInference, self).__init__(**kwargs)
        self.config = config
        if config.with_random_walks:
            try:
                concept2index['[CLS]']
            except KeyError:
                concept2index['[CLS]'] = len(concept2index)
        self.concept2index = concept2index
        self.index2concept = {v: k for k, v in concept2index.items()}
        self.fasttext = FastText(config)
        self.cnn = SqueezeExciteCNN(config, concept2index)#, valid_ids)
        print('CNN initialised')

    def call(self, inputs, training=False):
        text_embeddings = self.fasttext.batch_to_embeddings(inputs)
        cnn_logits = self.cnn(text_embeddings, training=training)
        return cnn_logits

    def predict_batch(self, batch):
        if isinstance(batch, str):
            batch = [batch]
        logits = self.call(batch)
        softmax = tf.nn.softmax(logits, axis=-1)
        idxs = tf.argmax(softmax, axis=-1).numpy()

        output = []
        for i in range(len(batch)):
            output.append((idxs[i], softmax[i, idxs[i]].numpy()))

        return output

