import json
import pickle
import re


import numpy as np
import regex as re2
import pandas as pd


from combine import ForInference

import os
import sys


class Annotator():
    """"""
    def __init__(self, config):
        self.config = config
        self.concept2index = pickle.load(
            open(f'{config.param_dir}/{config.model_name}/concept2index.pkl', "rb")
        )

        self.valid_ids = pickle.load(
            open(f'{config.param_dir}/{config.model_name}/valid_ids.pkl', "rb")
        )

        self.index2concept = {v: k for k, v in self.concept2index.items()}
        self.hpo_alt2id = pickle.load(
            open(config.dict_dir + '/hpo_alt2id.pkl', "rb")
        )
        if config.with_random_walks:
            try:
                self.concept2index['[CLS]']
            except KeyError:
                self.concept2index['[CLS]'] = len(self.concept2index)

        hpo_ids = set([v for k, v in self.concept2index.items()
                       if k.startswith('HP:')])
        self.hpo_ids = hpo_ids
        self.model = ForInference(config, self.concept2index)
        _ = self.model(["hello world"])
        self.model.cnn.load()


    @staticmethod
    def nonconsumesplitconcat(pat, content):
        outer = pat.split(content)
        inner = pat.findall(content) + ['']
        return [pair[0] + pair[1] for pair in zip(outer, inner)]


    def chunk_text(self, text):
        pat = re2.compile(r"(?<!\w\.\w.)(?![^(]*\))(?<![A-Z]\.)(?!\s+[a-z\[,])"
                          r"(?!\s+[0-9]{4})(?<![\w]{1}\.[\w]{1})"
                          r"(?<=\.|\?|\!\:)\s+|\p{Cc}+|\p{Cf}+")
        sentences = self.nonconsumesplitconcat(pat, text)
        pat2 = re2.compile(r'[\\\\/\r\n\t\-\(\)\_]')
        text_replaced = [pat2.sub(' ', line) for line in sentences]

        pat3 = re2.compile(r'[^\w\d ]')
        chunks_large = [(sen_num, chunk) for sen_num, chunks
                        in enumerate(text_replaced) for chunk in
                        self.nonconsumesplitconcat(pat3, chunks)]
        pat4 = re.compile(' ')
        tokens = [w for chunk in chunks_large for w in self.nonconsumesplitconcat(pat4, chunk[1])]
        # tokens = [w for chunk in chunks_large for w in
        #           self.nonconsumesplitconcat(pat2, chunk[1])]
        chunk_size_range = range(1, min(7, len(tokens) + 1))
        chunks = []
        start_indices = []
        end_indices = []
        for size in chunk_size_range:

            for i in range(len(tokens) - size + 1):
                chunk = ' '.join(tokens[i:i + size])
                start_idx = sum(len(tokens[i]) for i in range(i))
                chunks.append(chunk)
                start_indices.append(start_idx)
                end_indices.append(start_idx + len(chunk))
        return chunks, start_indices, end_indices

    def annotate_text(self, text, threshold=0.5):
        chunks, start_indices, end_indices = self.chunk_text(text)
        predictions = []
        batch_size=512
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_start_indices = start_indices[i:i+batch_size]
            batch_end_indices = end_indices[i:i+batch_size]
            batch_predictions = self.model.predict_batch(batch)

            for chunk_prediction, start_index, end_index, chunk in zip(
                    batch_predictions, batch_start_indices, batch_end_indices,
                    batch
            ):
                if chunk_prediction[0] in self.hpo_ids:
                    predictions.append(
                        (start_index, end_index, chunk_prediction[0],
                         chunk_prediction[1],
                         chunk)
                    )

        filtered = [prediction for prediction in predictions
                    if (prediction[3] > threshold and 
                    is_space_or_empty(prediction[4]) == False)]

        filtered = [(start, end, self.model.index2concept[i], f"{score:.3f}", 
            '"'+re.sub(' +', ' ', re.sub(r'"', r"''", chunk))+'"') for
            start, end, i, score, chunk in filtered]
        
        return filtered


def is_space_or_empty(s):
    stripped_s = s.replace('.', '').replace(',', '').strip()
    return stripped_s.isspace() or stripped_s == ''

def annotate_stream(model, threshold, input_iterator, output_writer):
    for i, (key, text) in enumerate(input_iterator):
        print("\rAnnotating file number %d" % i, end="")
        ants = model.annotate_text(text, threshold=threshold)
        output_writer.write(key, ants, model)
    sys.stdout.write("\n")


class DirOutputStream:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write(self, key, ants, model):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir + '/' + key, 'w') as fp:
            ants = sorted(ants, key=lambda x: x[0])
            for ant in ants:
                fp.write(
                    '\t'.join(map(str, ant)) + '\n')


class DirInputStream:

    def __init__(self, input_dir, filelist=None):
        self.input_dir = input_dir
        if filelist is None:
            self.filelist = os.listdir(self.input_dir)
        else:
            self.filelist = [x for x in os.listdir(self.input_dir)
                             if x in os.listdir(filelist)]

    def __len__(self):
        return len(self.filelist)

    def __iter__(self):
        self.filelist_iter = iter(self.filelist)
        return self

    def __next__(self):
        filename = next(self.filelist_iter)
        return filename, open(self.input_dir + '/' + filename,
                              encoding='utf-8').read()


class Config(object):
    def __init__(self, d):
        self.__dict__ = d



def macro_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def get_metrics(concept2index, model_output_dir, ground_truth_dir, alt2id):
    tp = [0] * len(concept2index)
    fp = [0] * len(concept2index)
    fn = [0] * len(concept2index)

    # loop over files in directory
    for filename in os.listdir(model_output_dir):

        # extract HPO IDs and scores from output file
        pred_file = os.path.join(model_output_dir, filename)

        try:
            y_pred = pd.read_csv(pred_file, sep='\t', header=None, usecols=[2], 
                                 engine='python', on_bad_lines='warn', 
                                 quotechar='"').values.tolist()
            # flatten list of lists
            y_pred = [item for sublist in y_pred for item in sublist]
        except pd.errors.EmptyDataError:
            continue
        except pd.errors.ParserError:
            raise Exception(f'ParserError at file {filename}')

        # extract HPO IDs from corresponding ground truth file
        truth_file = os.path.join(ground_truth_dir, filename)
        y_true = pd.read_csv(truth_file, sep='\t', header=None,  
                             usecols=[2]).values.tolist()
        # flatten list of lists
        y_true = [re.sub('_', ':', item)  for sublist in y_true for item in 
                  sublist]

        y_true = [alt2id.get(item, item) for item in y_true]
        y_true = [''.join(item) if isinstance(item, list) else item for item in 
                  y_true]

        # determine true positives, false positives, and false negatives
        document_tp = set(y_pred).intersection(set(y_true))
        document_fp = set(y_pred).difference(set(y_true))
        document_fn = set(y_true).difference(set(y_pred))

        # update counts
        for concept in document_tp:
            try: # Obsolete terms
                tp[concept2index[concept]] += 1
            except KeyError:
                pass
        for concept in document_fp:
            try:
                fp[concept2index[concept]] += 1
            except KeyError:
                pass
        for concept in document_fn:
            try:
                fn[concept2index[concept]] += 1
            except KeyError:
                pass

    # Calculate the micro-averaged precision, recall, and F1-score
    try:
        micro_precision = sum(tp) / (sum(tp) + sum(fp))
    except ZeroDivisionError:
        micro_precision = 0
    try:
        micro_recall = sum(tp) / (sum(tp) + sum(fn))
    except ZeroDivisionError:
        micro_recall = 0
    try:
        micro_f1 = 2 * micro_precision * micro_recall / (
                    micro_precision + micro_recall)
    except ZeroDivisionError:
        micro_f1 = 0
    # Get indices where either tf and fn are > 0
    indices = [i for i, (x, y, z) in enumerate(zip(tp, fp, fn)) 
               if x + y + z > 0]
    # Calculate the macro-averaged precision, recall, and F1-score
    macro_precision = np.mean([tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] > 0 
                               else 0 for i in indices])
    macro_recall = np.mean([tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 
                            else 0 for i in indices])
    macro_f1s = [
        2 * ((tp[i] / (tp[i] + fp[i])) * (tp[i] / (tp[i] + fn[i]))) / (
            (tp[i] / (tp[i] + fp[i])) + (tp[i] / (tp[i] + fn[i]))) 
        if ((tp[i] + fp[i]) > 0 and (tp[i] + fn[i]) > 0 and tp[i] > 0) 
        else 0 
        for i in indices
    ]
    macro_f1 = np.mean(macro_f1s)

    return micro_precision, micro_recall, micro_f1, macro_precision, \
        macro_recall, macro_f1

def main():

    # Loop through all models in their directories:
    # list all directories in directory

    thresh_df = pd.DataFrame(
        columns=['model', 'model_type',  'batch_size',  'learning_rate',
                 'concept_dim',
                 'num_filters', 'only_hpo', 'l2_strength', 'random_walks',
                 'num_hop', 'input_dropout', 'multi_task_losses', 'warm_up',
                 'threshold', 'macro_precision', 'macro_recall', 'macro_f1',
                 'micro_precision', 'micro_recall', 'micro_f1', 'sum_f1']
    )

    # load concept2index pickle file
    with open(f'dictionaries/concept2index.pkl', 'rb') as f:
        concept2index = pickle.load(f)
    with open(f'dictionaries/hpo_alt2id.pkl', 'rb') as f:
        alt2id = pickle.load(f)

    saved_model_dir = 'saved_models/arc'
    for dir in os.listdir(saved_model_dir):
        # if os.path.isdir(os.path.join(saved_model_dir, dir)):
        if dir.startswith('arc') and os.path.isdir(
                os.path.join(saved_model_dir, dir)):
            print("Model: ", dir)

            # Load in config file
            with open(f'{saved_model_dir}/{dir}/config.json', 'r') as fp:
                config = Config(json.load(fp))
            model_type = 'architecture'
            batch_size = config.batch_size
            learning_rate = config.learning_rate
            concept_dim = config.concept_dim
            num_filters = config.num_filters
            only_hpo = config.only_hpo
            l2_strength = config.l2_strength
            random_walks = config.with_random_walks
            num_hop = config.max_path_len
            input_dropout = config.input_dropout_rate
            warm_up = 20

            if random_walks:
                multi_task_losses = str(config.multi_task_losses)
            else:
                multi_task_losses = None

            annotator = Annotator(config)
            print("model loaded")
            input_stream = DirInputStream(input_dir= 'hpo_gsc/Text',
                                          filelist='hpo_gsc/threshold_40')

            for threshold in np.arange(0.1, 1.0, 0.1):
                print(f'Threshold: {threshold}')
                output_dir = f'{saved_model_dir}/{dir}/threshold/{threshold}'
                # make output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_stream = DirOutputStream(f"{output_dir}")
                print("streams ready")
                annotate_stream(annotator, threshold, input_stream, output_stream)
                micro_precision, micro_recall, micro_f1, \
                    macro_precision, macro_recall, macro_f1 = get_metrics(
                    annotator.concept2index, model_output_dir=output_dir,
                    ground_truth_dir='hpo_gsc/Formatted/threshold_40', alt2id=alt2id
                )
                new_row = pd.DataFrame(
                    {
                        'model': [dir],
                        'model_type': [model_type],
                        'batch_size': [batch_size],
                        'learning_rate': [learning_rate],
                        'concept_dim': [concept_dim],
                        'num_filters': [num_filters],
                        'only_hpo': [only_hpo],
                        'l2_strength': [l2_strength],
                        'random_walks': [random_walks],
                        'num_hop': [num_hop],
                        'input_dropout': [input_dropout],
                        'multi_task_losses': [multi_task_losses],
                        'warm_up': [warm_up],
                        'threshold': [threshold],
                        'macro_precision': [macro_precision],
                        'macro_recall': [macro_recall], 'macro_f1': [macro_f1],
                        'micro_precision': [micro_precision],
                        'micro_recall': [micro_recall], 'micro_f1': [micro_f1],
                        'sum_f1': [micro_f1 + macro_f1]
                    }
                )
                thresh_df = pd.concat([thresh_df, new_row], ignore_index=True)

    # evaluate each threshold's performance for each model
    # set all nan values to 0
    thresh_df = thresh_df.fillna(0)
    # group by model and find the best row for each model using sum_f1
    thresh_df.sum_f1 = pd.to_numeric(thresh_df.sum_f1, errors='coerce')
    best_thresh = thresh_df.groupby('model').apply(
        lambda x: x.loc[x['sum_f1'].idxmax()]
    )
    # create dictionary of model name to best threshold
    best_thresh_dict = best_thresh.set_index('model')['threshold'].to_dict()
    # save thresh_df to csv
    thresh_df.to_csv('arc_thresh_df.csv', index=False)
    del thresh_df

    final_df = pd.DataFrame(
        columns=['model',  'model_type', 'batch_size',  'learning_rate',
                 'concept_dim',
                 'num_filters', 'only_hpo', 'l2_strength', 'random_walks',
                 'num_hop', 'input_dropout', 'multi_task_losses', 'warm_up',
                 'threshold', 'macro_precision', 'macro_recall', 'macro_f1',
                 'micro_precision', 'micro_recall', 'micro_f1']
    )

    # then final annotation using the best threshold for each model
    for dir in os.listdir(f'{saved_model_dir}'):
        if dir.startswith('arc') and os.path.isdir(
                os.path.join(saved_model_dir, dir)):
            print("Model: ", dir)

            # Load in config file
            with open(f'{saved_model_dir}/{dir}/config.json', 'r') as fp:
                config = Config(json.load(fp))

            annotator = Annotator(config)
            print("model loaded")
            model_type = 'architecture'
            batch_size = config.batch_size
            learning_rate = config.learning_rate
            concept_dim = config.concept_dim
            num_filters = config.num_filters
            only_hpo = config.only_hpo
            l2_strength = config.l2_strength
            random_walks = config.with_random_walks
            # if random_walks:
            num_hop = config.max_path_len
            input_dropout = config.input_dropout_rate
            warm_up = 20

            if random_walks:
                multi_task_losses = str(config.multi_task_losses)
            else:
                multi_task_losses = None

            input_stream = DirInputStream(input_dir='hpo_gsc/Text',
                                          filelist='hpo_gsc/final_188')
            output_dir = f'{saved_model_dir}/{dir}/final_output'
            # make output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_stream = DirOutputStream(output_dir)

            print("streams ready")
            annotate_stream(annotator, best_thresh_dict[dir], input_stream, output_stream)
            micro_precision, micro_recall, micro_f1, \
                macro_precision, macro_recall, macro_f1 = get_metrics(
                annotator.concept2index, model_output_dir=output_dir,
                ground_truth_dir='hpo_gsc/Formatted/final_188', alt2id=alt2id
            )
            new_row = pd.DataFrame(
                {
                    'model': [dir],
                    'model_type': [model_type],
                    'batch_size': [batch_size],
                    'learning_rate': [learning_rate],
                    'concept_dim': [concept_dim],
                    'num_filters': [num_filters],
                    'only_hpo': [only_hpo],
                    'l2_strength': [l2_strength],
                    'random_walks': [random_walks],
                    'num_hop': [num_hop],
                    'input_dropout': [input_dropout],
                    'multi_task_losses': [multi_task_losses],
                    'warm_up': [warm_up],
                    'threshold': [best_thresh_dict[dir]],
                    'macro_precision': [macro_precision],
                    'macro_recall': [macro_recall], 'macro_f1': [macro_f1],
                    'micro_precision': [micro_precision],
                    'micro_recall': [micro_recall], 'micro_f1': [micro_f1]
                }
            )
            final_df = pd.concat([final_df, new_row], ignore_index=True)
    # set all nan values to 0
    final_df = final_df.fillna(0)
    # save final_df to csv
    final_df.to_csv('arc_final_df.csv', index=False)

if __name__ == "__main__":
    main()
