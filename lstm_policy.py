# coding: utf-8
import tensorflow as tf

from assert_shape import assert_shape
import train_policy


class Settings:
    # Data properties
    VARIABLE_NUM = 4
    MIN_VARIABLE_NUM = VARIABLE_NUM
    CLAUSE_SIZE = 2
    CLAUSE_NUM = 20
    MIN_CLAUSE_NUM = 1

    # Neural net
    EMBEDDING_SIZE = 64
    LSTM_STATE_SIZE = 64
    LSTM_LAYERS = 1

    SAT_HIDDEN_LAYERS = 0
    SAT_HIDDEN_LAYER_SIZE = 64
    POLICY_HIDDEN_LAYERS = 0
    POLICY_HIDDEN_LAYER_SIZE = 64

    LEARNING_RATE = 0.01

    POLICY_LOSS_WEIGHT = 1
    SAT_LOSS_WEIGHT = 1
    BATCH_SIZE = 64

    # Size of dataset

    SAMPLES = 10 ** 8

    # Multiprocessing
    PROCESSOR_NUM = None  # defaults to all processors

    BOARD_WRITE_GRAPH = False
    SR_GENERATOR = False

    NEPTUNE_ENABLED = False


DEFAULT_SETTINGS = Settings.__dict__.copy()


class Graph:
    def __init__(self, settings):
        BATCH_SIZE = settings["BATCH_SIZE"]
        CLAUSE_SIZE = settings["CLAUSE_SIZE"]
        VARIABLE_NUM = settings["VARIABLE_NUM"]
        EMBEDDING_SIZE = settings["EMBEDDING_SIZE"]
        LSTM_STATE_SIZE = settings["LSTM_STATE_SIZE"]
        LSTM_LAYERS = settings["LSTM_LAYERS"]
        POLICY_HIDDEN_LAYERS = settings["POLICY_HIDDEN_LAYERS"]
        POLICY_HIDDEN_LAYER_SIZE = settings["POLICY_HIDDEN_LAYER_SIZE"]
        SAT_HIDDEN_LAYER_SIZE = settings["SAT_HIDDEN_LAYER_SIZE"]
        SAT_LOSS_WEIGHT = settings["SAT_LOSS_WEIGHT"]
        POLICY_LOSS_WEIGHT = settings["POLICY_LOSS_WEIGHT"]

        self.inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, None, CLAUSE_SIZE), name='inputs')
        self.lengths = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='lengths')
        self.policy_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, VARIABLE_NUM,  2), name='policy_labels')
        self.sat_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE,), name='sat_labels')

        policy_labels_squeezed = tf.reshape(self.policy_labels, (BATCH_SIZE, VARIABLE_NUM * 2))

        vars_ = tf.abs(self.inputs)
        signs = tf.cast(tf.sign(self.inputs), tf.float32)  # shape: [batch_size, None, CLAUSE_SIZE]

        embeddings = tf.Variable(tf.random_uniform([VARIABLE_NUM + 1, EMBEDDING_SIZE], -1., 1), name='embeddings')

        var_embeddings = tf.nn.embedding_lookup(embeddings, vars_)
        # var_embeddings shape: [None, None, CLAUSE_SIZE, EMBEDDING_SIZE]
        
        clause_preembeddings = tf.concat(
            [tf.reshape(var_embeddings, [BATCH_SIZE, -1, CLAUSE_SIZE * EMBEDDING_SIZE]),
             signs],
            axis=2)
        
        PREEMBEDDING_SIZE = EMBEDDING_SIZE * CLAUSE_SIZE + CLAUSE_SIZE
        assert_shape(clause_preembeddings,
                     [BATCH_SIZE, None, PREEMBEDDING_SIZE])
        
        clause_w = tf.Variable(tf.random_normal(
            [PREEMBEDDING_SIZE, EMBEDDING_SIZE]), name='clause_w')
        clause_b = tf.Variable(tf.random_normal([EMBEDDING_SIZE]), name='clause_b')
        clause_embeddings = tf.reshape(tf.sigmoid(
            tf.reshape(clause_preembeddings, [-1, PREEMBEDDING_SIZE]) @ clause_w + clause_b), 
                                       [BATCH_SIZE, -1, EMBEDDING_SIZE])
        # shape: [None, None, EMBEDDING_SIZE]
        
        lstm = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(LSTM_STATE_SIZE)
             for i in range(LSTM_LAYERS)])
        
        _, lstm_final_states = tf.nn.dynamic_rnn(lstm, clause_embeddings, dtype=tf.float32,
                                                 sequence_length=self.lengths
                                               )
        formula_embedding = lstm_final_states[-1].h
            
        assert_shape(formula_embedding, [BATCH_SIZE, LSTM_STATE_SIZE])

        last_policy_layer = formula_embedding
        for num in range(POLICY_HIDDEN_LAYERS):
            last_policy_layer = tf.layers.dense(
                last_policy_layer, POLICY_HIDDEN_LAYER_SIZE, tf.nn.relu,
                name='policy_hidden_{}'.format(num + 1))
        self.policy_logits = tf.layers.dense(
            last_policy_layer, VARIABLE_NUM*2, name='policy')

        last_sat_layer = formula_embedding
        for num in range(POLICY_HIDDEN_LAYERS):
            last_sat_layer = tf.layers.dense(
                last_sat_layer, SAT_HIDDEN_LAYER_SIZE, tf.nn.relu,
                name='sat_hidden_{}'.format(num + 1))
        self.sat_logits = tf.squeeze(
            tf.layers.dense(last_sat_layer, 1, name='sat'), axis=1)
        
        # zero out policy for test when UNSAT
        # requires sat_labels to be provided, so needs to be a separate tensor in order 
        # for inference to work
        self.policy_logits_for_cmp = tf.expand_dims(self.sat_labels, axis=1) * self.policy_logits
        
        self.policy_loss = tf.losses.sigmoid_cross_entropy(
            policy_labels_squeezed, self.policy_logits_for_cmp)
        self.policy_probabilities = tf.sigmoid(self.policy_logits, name='policy_prob')
        self.policy_probabilities_for_cmp = tf.sigmoid(self.policy_logits_for_cmp)
        
        self.sat_loss = tf.losses.sigmoid_cross_entropy(self.sat_labels, self.sat_logits)
        self.sat_probabilities = tf.sigmoid(self.sat_logits, name='sat_prob')

        self.policy_top1_error = 1.0 - tf.reduce_sum(tf.gather_nd(
            policy_labels_squeezed,
            tf.stack([tf.range(BATCH_SIZE), tf.argmax(self.policy_probabilities_for_cmp, axis=1, output_type=tf.int32)],
                     axis=1))) / (tf.reduce_sum(self.sat_labels))

        self.policy_error = tf.reduce_sum(tf.abs(
            tf.round(self.policy_probabilities_for_cmp) - policy_labels_squeezed)) / (
              tf.reduce_sum(self.sat_labels)) / (VARIABLE_NUM * 2)
        self.sat_error = tf.reduce_mean(tf.abs(
            tf.round(self.sat_probabilities) - self.sat_labels))
        
        self.loss = SAT_LOSS_WEIGHT * self.sat_loss + POLICY_LOSS_WEIGHT * self.policy_loss
        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("policy_loss", self.policy_loss)
        tf.summary.scalar("policy_error", self.policy_error)
        tf.summary.scalar("policy_top1_error", self.policy_top1_error)
        tf.summary.scalar("sat_loss", self.sat_loss)
        tf.summary.scalar("sat_error", self.sat_error)
        tf.summary.scalar("sat_fraction", tf.reduce_sum(self.sat_labels) / BATCH_SIZE)


def main():
    train_policy.train_policy(Graph, DEFAULT_SETTINGS, representation="sequence")


if __name__ == "__main__":
    main()
