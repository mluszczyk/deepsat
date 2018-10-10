import tensorflow as tf

from assert_shape import assert_shape
import train_policy


class Settings:
    # Data properties
    SR_GENERATOR = False
    VARIABLE_NUM = 8
    MIN_VARIABLE_NUM = 8
    CLAUSE_SIZE = 3
    CLAUSE_NUM = 40
    MIN_CLAUSE_NUM = 1

    # Neural net
    EMBEDDING_SIZE = 128
    LEVEL_NUMBER = 10

    POS_NEG_ACTIVATION = None
    HIDDEN_LAYERS = [128, 128]
    HIDDEN_ACTIVATION = tf.nn.relu
    EMBED_ACTIVATION = None

    LEARNING_RATE = 0.001

    POLICY_LOSS_WEIGHT = 1
    SAT_LOSS_WEIGHT = 1
    BATCH_SIZE = 64

    NEPTUNE_ENABLED = False
    BOARD_WRITE_GRAPH = True

    # Size of dataset

    SAMPLES = 10 ** 8

    # Multiprocessing
    PROCESSOR_NUM = None  # defaults to all processors


DEFAULT_SETTINGS = Settings.__dict__.copy()


class Graph:
    def __init__(self, settings):
        BATCH_SIZE = None
        self.inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, None, 2), name='inputs')
        self.policy_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, 2), name='policy_labels')
        self.sat_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE,), name='sat_labels')

        batch_size = tf.shape(self.inputs)[0]
        variable_num = tf.shape(self.inputs)[1]
        clause_num = tf.shape(self.inputs)[2]
        assert_shape(variable_num, [])
        assert_shape(clause_num, [])

        variables_per_clause = tf.reduce_sum(self.inputs, axis=[1, 3])
        clauses_per_variable = tf.reduce_sum(self.inputs, axis=[2, 3])
        assert_shape(variables_per_clause, [BATCH_SIZE, None])
        assert_shape(clauses_per_variable, [BATCH_SIZE, None])

        positive_connections, negative_connections = (tf.squeeze(conn, axis=3) for conn in tf.split(self.inputs, 2, axis=3))
        assert_shape(positive_connections, [BATCH_SIZE, None, None])
        assert_shape(negative_connections, [BATCH_SIZE, None, None])

        reuse = tf.AUTO_REUSE

        self.loss = 0.0

        EMBEDDING_SIZE = settings["EMBEDDING_SIZE"]
        HIDDEN_LAYERS = settings["HIDDEN_LAYERS"]
        HIDDEN_ACTIVATION = settings["HIDDEN_ACTIVATION"]
        SAT_LOSS_WEIGHT = settings["SAT_LOSS_WEIGHT"]
        POLICY_LOSS_WEIGHT = settings["POLICY_LOSS_WEIGHT"]
        LEVEL_NUMBER = settings["LEVEL_NUMBER"]
        EMBED_ACTIVATION = settings["EMBED_ACTIVATION"]
        POS_NEG_ACTIVATION = settings["POS_NEG_ACTIVATION"]

        for level in range(LEVEL_NUMBER+1):
            if level == 0:
                initial_var_embedding = tf.Variable(
                    tf.random_uniform([EMBEDDING_SIZE], -1., 1),
                    name='init_var_embedding')
                var_embeddings = tf.tile(
                    tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
                    [batch_size, variable_num, 1])

                initial_clause_embedding = tf.Variable(
                    tf.random_uniform([EMBEDDING_SIZE], -1., 1),
                    name='init_clause_embedding')
                clause_embeddings = tf.tile(
                    tf.reshape(initial_clause_embedding, [1, 1, EMBEDDING_SIZE]),
                    [batch_size, clause_num, 1])

            elif level >= 1:
                positive_var_embeddings = tf.layers.dense(
                    var_embeddings, EMBEDDING_SIZE, activation=POS_NEG_ACTIVATION, name='positive_var', reuse=reuse)
                negative_var_embeddings = tf.layers.dense(
                    var_embeddings, EMBEDDING_SIZE, activation=POS_NEG_ACTIVATION, name='negative_var', reuse=reuse)
                assert_shape(positive_var_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
                assert_shape(negative_var_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                clause_preembeddings = tf.divide((
                    tf.matmul(positive_connections, positive_var_embeddings, transpose_a=True) +
                    tf.matmul(negative_connections, negative_var_embeddings, transpose_a=True)
                ), tf.expand_dims(variables_per_clause, -1) + 1.0) + clause_embeddings
                assert_shape(clause_preembeddings,
                             [BATCH_SIZE, None, EMBEDDING_SIZE])
                last_hidden = clause_preembeddings
                for i in HIDDEN_LAYERS:
                    last_hidden = tf.layers.dense(
                        last_hidden, EMBEDDING_SIZE,
                        activation=HIDDEN_ACTIVATION, name='hidden_clause_{}'.format(i),
                        reuse=reuse)
                clause_embeddings = tf.layers.dense(
                    last_hidden, EMBEDDING_SIZE, activation=EMBED_ACTIVATION, name='clause_embeddings', reuse=reuse)
                assert_shape(clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                # clause -> var

                positive_clause_embeddings = tf.layers.dense(
                    clause_embeddings, EMBEDDING_SIZE, activation=POS_NEG_ACTIVATION, name='positive_clause', reuse=reuse)
                negative_clause_embeddings = tf.layers.dense(
                    clause_embeddings, EMBEDDING_SIZE, activation=POS_NEG_ACTIVATION, name='negative_clause', reuse=reuse)
                assert_shape(positive_clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
                assert_shape(negative_clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                var_preembeddings = tf.divide((
                        tf.matmul(positive_connections, positive_clause_embeddings) +
                        tf.matmul(negative_connections, negative_clause_embeddings)
                ), tf.expand_dims(clauses_per_variable, -1) + 1.0) + var_embeddings
                assert_shape(var_preembeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
                last_hidden = var_preembeddings
                for i in HIDDEN_LAYERS:
                    last_hidden = tf.layers.dense(
                        last_hidden, EMBEDDING_SIZE,
                        activation=HIDDEN_ACTIVATION,
                        name='hidden_var_{}'.format(i),
                        reuse=reuse)
                var_embeddings = tf.layers.dense(
                    last_hidden, EMBEDDING_SIZE, activation=EMBED_ACTIVATION, name='var_embeddings', reuse=reuse)
            assert_shape(var_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

            self.policy_logits = tf.layers.dense(
                var_embeddings, 2, name='policy', reuse=reuse)
            assert_shape(self.policy_logits, [BATCH_SIZE, None, 2])

            self.sat_logits = tf.reduce_sum(
                tf.layers.dense(var_embeddings, 1, name='sat', reuse=reuse),
                axis=[1, 2])
            assert_shape(self.sat_logits, [BATCH_SIZE])

            # zero out policy for test when UNSAT
            # requires sat_labels to be provided, so needs to be a separate tensor in order
            # for inference to work
            self.policy_logits_for_cmp = tf.reshape(self.sat_labels, [batch_size, 1, 1]) * self.policy_logits

            self.policy_loss = tf.losses.sigmoid_cross_entropy(
                self.policy_labels, self.policy_logits_for_cmp)
            self.policy_probabilities = tf.sigmoid(self.policy_logits, name='policy_prob')
            self.policy_probabilities_for_cmp = tf.sigmoid(self.policy_logits_for_cmp)

            self.sat_loss = tf.losses.sigmoid_cross_entropy(self.sat_labels, self.sat_logits)
            self.sat_probabilities = tf.sigmoid(self.sat_logits, name='sat_prob')

            '''
            self.policy_top1_error = 1.0 - tf.reduce_sum(tf.gather_nd(
                self.policy_labels,
                tf.stack([tf.range(BATCH_SIZE), tf.argmax(self.policy_probabilities_for_cmp, axis=1, output_type=tf.int32)],
                         axis=1))) / (tf.reduce_sum(self.sat_labels))
            '''

            self.policy_error = tf.reduce_sum(tf.abs(
                tf.round(self.policy_probabilities_for_cmp) - self.policy_labels)) / (
                  tf.reduce_sum(self.sat_labels)) / (tf.cast(variable_num, dtype=tf.float32) * 2.0)
            self.sat_error = tf.reduce_mean(tf.abs(
                tf.round(self.sat_probabilities) - self.sat_labels))
        
            self.level_loss = SAT_LOSS_WEIGHT * self.sat_loss + POLICY_LOSS_WEIGHT * self.policy_loss
            self.loss += self.level_loss

            lvln = "_level_{}".format(level)
            tf.summary.scalar("loss"+lvln, self.level_loss)
            tf.summary.scalar("policy_loss"+lvln, self.policy_loss)
            tf.summary.scalar("policy_error"+lvln, self.policy_error)
            #tf.summary.scalar("policy_top1_error"+lvln, self.policy_top1_error)
            tf.summary.scalar("sat_loss"+lvln, self.sat_loss)
            tf.summary.scalar("sat_error"+lvln, self.sat_error)
            tf.summary.scalar("sat_fraction"+lvln, tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("policy_loss", self.policy_loss)
        tf.summary.scalar("policy_error", self.policy_error)
        # tf.summary.scalar("policy_top1_error", self.policy_top1_error)
        tf.summary.scalar("sat_loss", self.sat_loss)
        tf.summary.scalar("sat_error", self.sat_error)
        tf.summary.scalar("sat_fraction",
                          tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))


def main():
    train_policy.train_policy(Graph, DEFAULT_SETTINGS, representation='graph')


if __name__ == "__main__":
    main()
