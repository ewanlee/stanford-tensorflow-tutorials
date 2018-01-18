import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from itertools import tee
from process_data import read_data, build_vocab, convert_words_to_index

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

VOCAB_SIZE = 10000
EMBED_SIZE = 128

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class ContextCountModel:
    """ Build the graph for the context-count model """
    def __init__(self, file_path, vocab_size, embed_size):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def _preprocess(self):
        """ Read in data and build the vocabulary """
        words = read_data(self.file_path)
        self.dictionary, self.invert_dict = build_vocab(words, self.vocab_size)
        self.index_words = convert_words_to_index(words, self.dictionary)

    def _build_cocur_matrix(self):
        """ Build the con-occurrence matrix """
        cocur_matrix = np.zeros([self.vocab_size, self.vocab_size])
        for i, j in pairwise(self.index_words):
            cocur_matrix[i, j] += 1
        self.cocur_matrix = cocur_matrix + cocur_matrix.T - np.diag(
                cocur_matrix.diagonal())

    def _create_placeholder(self):
        with tf.device('/gpu:0'):
            with tf.name_scope('input'):
                self.cocur_matrix_ph = tf.placeholder(
                        tf.float32,
                        shape=[self.vocab_size, self.vocab_size],
                        name='cocur_matrix_ph')

    def _embedding(self):
        """ Calculate the embedding matrix  """
        with tf.device('/gpu:0'):
            with tf.name_scope('embed'):
                _, u, _ = tf.svd(self.cocur_matrix_ph, name='svd')
                self.embedding_matrix = tf.slice(u, [0, 0],
                        size=[self.vocab_size, self.embed_size],
                        name='embedding')

    def build_graph(self):
        """ Build the graph for this model """
        self._preprocess()
        self._build_cocur_matrix()
        self._create_placeholder()
        self._embedding()


def train_model(model):
    co_occur_mat = model.cocur_matrix
    print(co_occur_mat[:10, :10])
    with tf.Session() as sess:
        embedding_matrix = sess.run(
                model.embedding_matrix,
                feed_dict={model.cocur_matrix_ph: co_occur_mat})

        # code to visualize the embeddings.
        # run "'tensorboard --logdir='processed'" to see the embeddings

        # it has to variable.
        # constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(embedding_matrix[:1000],
                name='visible_embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file,
        # in this case the first 1000 words of vocab
        embedding.metadata_path = 'processed/vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model_count.ckpt', 1)

    return embedding_matrix

def main():
    file_path = './data/count/text8.zip'
    model = ContextCountModel(file_path, VOCAB_SIZE, EMBED_SIZE)
    model.build_graph()
    invert_dict = model.invert_dict
    embedding_matrix = train_model(model)
    print(embedding_matrix[:10])


if __name__ == '__main__':
    main()
