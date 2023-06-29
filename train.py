import os

# UNCOMMENT FOR GPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pickle as pkl
import tensorflow as tf
import pandas as pd

from utils import TrainingArgs, TrainingCounter
from stream_text import text_input_function, FastText
from combine import ForTraining
from valid_walk import OutputLayer
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
    data_gen = text_input_function(
        config=config,
        total_phrase_num=num_samples,
        label_path=f'{config.data_dir}/{config.model_name}/labels.txt',
        data_path=f'{config.data_dir}/{config.model_name}/texts.txt'
    )
    print('Data generator initialised')
    counter = TrainingCounter(config=config, total_phrase_num=num_samples)
    print('Training counter initialised')
    
    # if config.with_random_walks:
    warm_up = 15*(num_samples // config.batch_size)
    lr_schedule = LinearWarmup(warmup_steps=warm_up,
                               final_learning_rate=config.learning_rate)
    optimiser = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if config.with_random_walks:
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_history = pd.DataFrame(
            columns=['epoch', 'loss', 'walk_test_loss', 'walk_test_acc']
        )
    else:
        loss_history = pd.DataFrame(columns=['epoch', 'loss'])

    print('Optimiser initialised')
    fasttext = FastText(config=config)
    print('FastText initialised')

    model = ForTraining(config=config,
                        relation_dicts=relation_dicts,
                        concept2index=concept2index)
                        # valid_ids=valid_ids)
    print('Model initialised')

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

    dropout_change = 0
    first_pass = True
    print('Starting training')
    repeat_count = (len(concept2index) +
                    config.max_neighbourhood_sample - 1
                    ) // (config.max_neighbourhood_sample)
    # to monitor how accurate the walk classifier is during the training.

    for batch in data_gen:
        labels = batch[0]
        text_embeddings = fasttext.batch_to_embeddings(batch[1])
        check_for_nan_or_large_values(text_embeddings)
        with tf.GradientTape() as tape:

            if config.with_random_walks:
                cnn_out, gnn_out, walk_labels, walk_predicted = model(
                    [labels, text_embeddings], training=True
                )

                logits = tf.einsum('bi,bji->bj', cnn_out, gnn_out)
                padding_size = len(
                    concept2index) - config.max_neighbourhood_sample
                padding = tf.zeros([tf.shape(logits)[0], padding_size],
                                   dtype=logits.dtype) + 0.5
                logits = tf.concat(
                    [logits[:, :config.max_neighbourhood_sample], padding],
                    axis=-1)
                cnn_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.zeros([logits.shape[0]], dtype=tf.int32),
                        logits=logits
                    )
                )
                gnn_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=walk_labels, logits=walk_predicted
                    )
                )
                if first_pass:
                    # calculate loss scaling based on initial loss
                    first_pass = False
                    initial_total_loss = cnn_loss + gnn_loss
                    cnn_target_ratio = config.multi_task_losses[0]
                    gnn_target_ratio = config.multi_task_losses[1]
                    cnn_scaling_factor = (initial_total_loss * cnn_target_ratio) / cnn_loss
                    gnn_scaling_factor = (initial_total_loss * gnn_target_ratio) / gnn_loss
                loss = cnn_loss * cnn_scaling_factor + gnn_loss * gnn_scaling_factor
            else:
                cnn_out, gnn_out = model([labels, text_embeddings], training=True)
                logits = tf.einsum('bi,bji->bj', cnn_out, gnn_out)
                padding_size = len(
                    concept2index) - config.max_neighbourhood_sample
                padding = tf.zeros([tf.shape(logits)[0], padding_size],
                                   dtype=logits.dtype) + 0.5
                logits = tf.concat(
                    [logits[:, :config.max_neighbourhood_sample], padding],
                    axis=-1)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.zeros([logits.shape[0]], dtype=tf.int32),
                        logits=logits
                    )
                )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        loss_value = loss.numpy()
        counter(loss=loss_value)

        if counter.epoch_end:
            if counter.epoch == 1:
                model.gnn.plot_kernel_weights_lattice(name='after_1_epoch')
            if counter.no_improvement == 0:
                model.save()
            if config.with_random_walks:
                # produce a test batch
                test_labels, walks, adj_mat, degree = \
                    model.adjacency.get_batch_random_walks(
                        batch_size=config.batch_size
                    )
                node_embeddings = model.cnn.node_embeddings(walks)
                meta_path_embeddings = model.gnn(
                    [adj_mat, degree, node_embeddings],
                    training=False)
                test_logits = model.output_layer(meta_path_embeddings,
                                                 training=False)
                gnn_test_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=test_labels, logits=test_logits
                    )
                )
                # Compute predicted labels
                predicted_labels = tf.math.round(tf.nn.sigmoid(test_logits))
                # Compute accuracy
                accuracy = accuracy_object(test_labels, predicted_labels)
                print(f'Test loss: {gnn_test_loss:.3f}, '
                      f'Test accuracy: {accuracy:.3f}')
                new_row = pd.DataFrame(
                    {'epoch': [counter.epoch], 'loss': [loss_value]}
                )
            else:
                new_row = pd.DataFrame(
                    {'epoch': [counter.epoch], 'loss': [loss_value]}
                )
            loss_history = pd.concat([loss_history, new_row],
                                     ignore_index=True)

            if counter.stop:
                if lr_schedule.step > warm_up:
                    print(f"Early stopping at epoch {counter.epoch}")
                    break

    print('Training complete')
    model.load()  # Get best weights
    model.replace_embeddings()
    with open(f'{config.param_dir}/{config.model_name}/config.json', 'w') as handle:
        json.dump(config.__dict__, handle, indent=4)
    model.gnn.plot_kernel_weights_lattice(name='final')
    loss_history.to_csv(os.path.join(f'{config.param_dir}/{config.model_name}',
                                     'training_loss.csv'),
                        index=False)

if __name__ == '__main__':
    main()

