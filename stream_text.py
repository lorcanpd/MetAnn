
import re

import tensorflow as tf
import fasttext as ft


class FastText(object):

    def __init__(self, config, max_len=16, embedding_dim=200):
        # fasttext embeddings from https://github.com/ncbi-nlp/BioSentVec
        self.model = ft.load_model(f'{config.fasttext}/BioWordVec_PubMed_MIMICIII_d200.bin')
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.pattern = re.compile('[\W_]')

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def tokenize(self, phrase):
        tmp = self.pattern.sub(' ', phrase).lower().strip().split(' ')
        output = ['INT' if w.isdigit() 
                  else ('FLOAT' if self.is_number(w) else w) 
                  for w in tmp]
        return output

    def sentence_to_embeddings(self, sentence):
        # decode tokens to strings
        words = [w.decode() if isinstance(w, bytes) else w
                 for w in self.tokenize(sentence)][:self.max_len]
        embeddings = []
        
        for i in range(self.max_len):
            try:
                embeddings.append(self.model.get_word_vector(words[i]))
            except IndexError:
                embeddings.append(np.zeros(self.embedding_dim))

        return tf.stack(embeddings)

    def batch_to_embeddings(self, batch_sentences):
        batch_embeddings = []
        for i, sentence in enumerate(batch_sentences):
            if isinstance(sentence, tf.Tensor):
                sentence = sentence.numpy().decode()
            if isinstance(sentence, bytes):
                sentence = sentence.decode()

            embeddings = self.sentence_to_embeddings(sentence)
            batch_embeddings.append(embeddings)
        return tf.stack(batch_embeddings)


def text_data_generator(label_path, data_path):
    with open(label_path, 'r') as labels, open(data_path, 'r') as data:
        for label, phrase in zip(labels, data):
            label =  int(label)
            yield label, phrase.strip()

# stream labels and text from files using tf.data.Dataset.from_generator
def text_input_function(config, total_phrase_num, label_path, data_path):

    types = (tf.int32, tf.string)

    dataset = tf.data.Dataset.from_generator(
        generator=text_data_generator,
        output_types=types,
        args=(label_path, data_path)
    )
    if config.shuffle and config.repeat:
        dataset = (dataset.shuffle(buffer_size=total_phrase_num)
                   .batch(batch_size=config.batch_size)
                   .repeat(config.epochs))
    elif config.shuffle and not config.repeat:
        dataset = (dataset.shuffle(buffer_size=total_phrase_num)
                   .batch(batch_size=config.batch_size))
    elif config.repeat and not config.shuffle:
        dataset = (dataset.batch(batch_size=config.batch_size)
                   .repeat(config.epochs))
    else:
        dataset = dataset.batch(batch_size=config.batch_size)
    return dataset


