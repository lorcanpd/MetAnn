import os

# # UNCOMMENT FOR GPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pickle as pkl
import tensorflow as tf
import pandas as pd

from utils import TrainingArgs, TrainingCounter
from stream_text import text_input_function, FastText

from valid_walk import OutputLayer
from gnn import MetaPathTransformer
from ner_model import SqueezeExciteCNN
from adjacency import Adjacency
from learning_rate import LinearWarmup

import argparse
import json

def main():

    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--params',
                        help="json file containing training parameters")
    config = TrainingArgs(parser.parse_args().params)
    # If param_dir does not exist, create it
    try:
        os.mkdir(config.param_dir)
    except FileExistsError:
        pass
    # If data_dir does not exist, create it
    try:
        os.mkdir(config.data_dir)
    except FileExistsError:
        pass
    # If figures_dir does not exist, create it
    try:
        os.mkdir(config.figures_dir)
    except FileExistsError:
        pass

    with open(f'{config.dict_dir}/relation_dicts.pkl', 'rb') as handle:
        relation_dicts = pkl.load(handle)
    with open(f'{config.dict_dir}/concept2index.pkl', 'rb') as handle:
        concept2index = pkl.load(handle)
    with open(f'{config.dict_dir}/index2concept.pkl', 'rb') as handle:
        index2concept = pkl.load(handle)


    # use name dictionaries to create label and text txt files to stream data from
    #  for training
    with open(f'{config.dict_dir}/hpo_names.pkl', 'rb') \
            as handle:
        hpo_names = pkl.load(handle)
    with open(f'{config.dict_dir}/gene_names.pkl', 'rb') \
            as handle:
        gene_names = pkl.load(handle)

    valid_ids = set()
    all_names = {}
    for k, v in hpo_names.items():
        idx = concept2index[k]
        valid_ids.add(idx)
        all_names[idx] = v

    if not config.only_hpo:
        for k, v in gene_names.items():
            idx = concept2index[k]
            valid_ids.add(idx)
            all_names[idx] = v

    concept2index['<PAD>'] = 0
    index2concept[0] = '<PAD>'
    try:
        concept2index['[CLS]']
    except KeyError:
        concept2index['[CLS]'] = len(concept2index)

    try:
        os.mkdir(f'{config.figures_dir}/{config.model_name}')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{config.param_dir}/{config.model_name}')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{config.data_dir}/{config.model_name}')
    except FileExistsError:
        pass

    with open(f'{config.data_dir}/{config.model_name}/labels.txt', 'w') as labels, \
            open(f'{config.data_dir}/{config.model_name}/texts.txt', 'w') as texts:
        for idx, v in all_names.items():
            for name in v:
                texts.write(f'{name}\n')
                labels.write(f'{idx}\n')

    # pickle the for_mask set
    with open(f'{config.param_dir}/{config.model_name}/valid_ids.pkl', 'wb') as handle:
        pkl.dump(valid_ids, handle)

    # pickle model specific concept2oindex.
    with open(f'{config.param_dir}/{config.model_name}/concept2index.pkl', 'wb') as handle:
        pkl.dump(concept2index, handle)

    print('Text and label files created')
    # efficiently count the number of lines in a text file
    num_samples = sum(1 for line in open(f'{config.data_dir}/{config.model_name}/labels.txt'))
    # create datagen object to stream data from text and label files

    text_counter = TrainingCounter(config=config, total_phrase_num=num_samples)
    walk_counter = TrainingCounter(config=config, total_phrase_num=num_samples)
    print('Training counters initialised')
    walk_loss_history = pd.DataFrame(
        columns=['epoch', 'training_loss', 'test_loss', 'test_acc']
    )
    loss_history = pd.DataFrame(
        columns=['epoch', 'loss', 'walk_test_loss', 'walk_test_acc']
    )
    warm_up = 20*(num_samples // config.batch_size) # 15 epoch warm up
    walk_lr_schedule = LinearWarmup(warmup_steps=warm_up,
                                    final_learning_rate=config.learning_rate)
    text_lr_schedule = LinearWarmup(warmup_steps=warm_up,
                                    final_learning_rate=config.learning_rate)
    gnn_optimiser = tf.keras.optimizers.Adam(learning_rate=walk_lr_schedule)
    # Make learning rate for CNN higher as its learning is more stable.
    cnn_optimiser = tf.keras.optimizers.Adam(
        learning_rate=text_lr_schedule
    )

    print('Optimiser initialised')
    fasttext = FastText(config=config)
    print('FastText initialised')

    # Initialise models.
    text_cnn = SqueezeExciteCNN(config=config, concept2index=concept2index)
    meta_path = MetaPathTransformer(config=config)
    gnn_output = OutputLayer(config=config)
    adjacency = Adjacency(relation_dicts, concept2index, config)
    print('Models initialised')

    def check_for_nan_or_large_values(vectors):
        for i, vector in enumerate(vectors):
            try:
                vector = tf.debugging.check_numerics(
                    vector,
                    message=f"Vector {i} contains NaN or infinite values."
                )
            except Exception as e:
                print(str(e))

            large_values = tf.reduce_max(tf.abs(vector))
            if large_values > 1e5:
                print(f"Vector {i} has a very large magnitude.")

    # Pre-training using walk validity task.
    accuracy_object = tf.keras.metrics.BinaryAccuracy()
    print('\nStarting pre-training using walk validity task')
    # Make it match sample size, total epochs and batch size of text classifier
    # training.
    for _ in range((config.epochs * num_samples // config.batch_size)+1):
        with tf.GradientTape() as tape:
            labels, walks, adj_mat, degree = adjacency.get_batch_random_walks(
                batch_size=config.batch_size
            )
            node_embeddings = text_cnn.node_embeddings(walks)
            meta_path_embeddings = meta_path(
                [adj_mat, degree, node_embeddings],
                training=True
            )
            logits = gnn_output(meta_path_embeddings, training=True)
            gnn_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )
        gradients = tape.gradient(
            gnn_loss,
            meta_path.trainable_variables +
            gnn_output.trainable_variables +
            text_cnn.node_embeddings.trainable_variables
        )
        gnn_optimiser.apply_gradients(
            zip(gradients,
                meta_path.trainable_variables +
                gnn_output.trainable_variables +
                text_cnn.node_embeddings.trainable_variables)
        )
        loss_value = gnn_loss.numpy()
        epoch_loss = walk_counter.epoch_loss
        walk_counter(loss=loss_value)

        # End of epoch.
        if walk_counter.epoch_end:

            # produce a test batch
            labels, walks, adj_mat, degree = adjacency.get_batch_random_walks(
                batch_size=config.batch_size
            )
            node_embeddings = text_cnn.node_embeddings(walks)
            meta_path_embeddings = meta_path([adj_mat, degree, node_embeddings],
                                                training=False)
            logits = gnn_output(meta_path_embeddings, training=False)
            gnn_test_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )

            # Compute predicted labels
            predicted_labels = tf.math.round(tf.nn.sigmoid(logits))

            # Compute accuracy
            accuracy = accuracy_object(labels, predicted_labels)
            print(f'Test loss: {gnn_test_loss:.3f}, '
                  f'Test accuracy: {accuracy:.3f}')
            new_row = pd.DataFrame(
                dict(epoch=[walk_counter.epoch],
                     training_loss=[epoch_loss],
                     test_loss=[gnn_test_loss],
                     test_acc=[accuracy])
            )
            walk_loss_history = pd.concat([walk_loss_history, new_row])

            if walk_counter.epoch == 1:
                meta_path.plot_kernel_weights_lattice(name='after_1_epoch')

            # If early stopping is triggered, break out of training loop.
            if walk_counter.stop:
                if walk_lr_schedule.step > warm_up:

                    print('Early stopping triggered')
                    break
    walk_loss_history.to_csv(
        f'{config.param_dir}/{config.model_name}/walk_loss_history.csv',
        index=False
    )
    meta_path.plot_kernel_weights_lattice(name='after_pretraining')
    gnn_output.save()
    print('Pre-training using walk validity task complete')

    data_gen = text_input_function(
        config=config,
        total_phrase_num=num_samples,
        label_path=f'{config.data_dir}/{config.model_name}/labels.txt',
        data_path=f'{config.data_dir}/{config.model_name}/texts.txt'
    )
    print('\nData generator for text classifier initialised')

    # Text classifier training.
    print('Starting text classifier training')
    for batch in data_gen:
        labels = batch[0]
        text_embeddings = fasttext.batch_to_embeddings(batch[1])
        check_for_nan_or_large_values(text_embeddings)
        with tf.GradientTape() as tape:
            cnn_out = text_cnn(text_embeddings, training=True)
            node_samples = adjacency.batch_nodes_within_n_hops(
                labels, config.max_path_len, training=True
            )
            padded_node_samples = adjacency.pad_node_list_batch(
                node_samples
            )
            node_embeddings = text_cnn.node_embeddings(padded_node_samples)
            adj_mat, degree = adjacency.batch_adjacency_matrices(
                node_samples)

            gnn_out = meta_path([adj_mat, degree, node_embeddings],
                                training=True)

            logits = tf.einsum('bi,bji->bj', cnn_out, gnn_out)

            padding_size = len(
                concept2index) - config.max_neighbourhood_sample
            padding = tf.zeros([tf.shape(logits)[0], padding_size],
                               dtype=logits.dtype) + 0.5
            # Concatenating the original logits with the padding
            logits = tf.concat(
                [logits[:, :config.max_neighbourhood_sample], padding], axis=-1)
            cnn_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([logits.shape[0]], dtype=tf.int32),
                    logits=logits
                )
            )
        gradients = tape.gradient(
            cnn_loss,
            text_cnn.trainable_variables + meta_path.trainable_variables
        )
        cnn_optimiser.apply_gradients(
            zip(gradients,
                text_cnn.trainable_variables + meta_path.trainable_variables)
        )
        loss_value = cnn_loss.numpy()

        epoch_loss = text_counter.epoch_loss
        text_counter(loss=loss_value)

        # End of epoch.
        if text_counter.epoch_end:
            if text_counter.no_improvement == 0:
                text_cnn.save()
                meta_path.save()


            # produce a test batch
            labels, walks, adj_mat, degree = adjacency.get_batch_random_walks(
                batch_size=config.batch_size
            )
            node_embeddings = text_cnn.node_embeddings(walks)
            meta_path_embeddings = meta_path([adj_mat, degree, node_embeddings],
                                                training=False)
            logits = gnn_output(meta_path_embeddings, training=False)
            gnn_test_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )

            # Compute predicted labels
            predicted_labels = tf.math.round(tf.nn.sigmoid(logits))

            # Compute accuracy
            accuracy = accuracy_object(labels, predicted_labels)
            print(f'Walk validity model test loss: {gnn_test_loss:.3f}, and '
                  f'Test accuracy: {accuracy:.3f}')

            new_row = pd.DataFrame(
                dict(epoch=[text_counter.epoch],
                     loss=[epoch_loss],
                     walk_test_loss=[gnn_test_loss],
                     walk_test_acc=[accuracy])
            )
            loss_history = pd.concat([loss_history, new_row])


            # If early stopping is triggered, break out of training loop.
            if text_counter.stop:
                if text_lr_schedule.step > warm_up:
                    break
    loss_history.to_csv(
        f'{config.param_dir}/{config.model_name}/loss_history.csv',
        index=False
    )
    print('Text classifier training complete')

    text_cnn.load() # Load best model
    meta_path.load() # Load best model

    meta_path.plot_kernel_weights_lattice(name='final')
    num_embeddings = text_cnn.node_embeddings.embeddings.shape[0]
    new_embeddings_list = []

    print('Replacing embeddings with aggregated embeddings from GNN output')

    for i in range(0, num_embeddings, config.batch_size):
        batch_indices = [
            x for x in
            range(i, min(i + config.batch_size, num_embeddings))
        ]
        neighbourhood = adjacency.batch_nodes_within_n_hops(
            batch_indices, config.max_path_len, training=False
        )
        padded_neighbourhood = adjacency.pad_node_list_batch(
            neighbourhood
        )
        adj_mat, degree = adjacency.batch_adjacency_matrices(
            neighbourhood
        )
        node_embeddings = text_cnn.node_embeddings(padded_neighbourhood)
        new_embeddings_list.extend(
            meta_path([adj_mat, degree, node_embeddings])[:, 0, :].numpy()
        )
    new_embeddings = tf.convert_to_tensor(
        new_embeddings_list, dtype=text_cnn.node_embeddings.embeddings.dtype
    )
    text_cnn.local_node_embeddings = text_cnn.node_embeddings
    text_cnn.node_embeddings.set_weights([new_embeddings])
    text_cnn.save()

    with open(f'{config.param_dir}/{config.model_name}/config.json', 'w') as handle:
        json.dump(config.__dict__, handle, indent=4)


if __name__ == '__main__':
    main()

