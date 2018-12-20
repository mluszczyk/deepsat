# coding: utf-8

import tensorflow as tf
import numpy as np


tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_string("train_file", None, "Train file")


FLAGS = tf.flags.FLAGS


DEFAULT_SETTINGS = {

    # Only for not SR
    "MIN_CLAUSE_NUM": 1,

    "SR_GENERATOR": True,

    # Neural net
    "EMBEDDING_SIZE": 128,

    "POS_NEG_ACTIVATION": None,
    "HIDDEN_LAYERS": [128, 128],
    "HIDDEN_ACTIVATION": tf.nn.relu,
    "EMBED_ACTIVATION": tf.nn.tanh,


    "POLICY_LOSS_WEIGHT": 1,
    "SAT_LOSS_WEIGHT": 1,

    "NEPTUNE_ENABLED": False,
    "BOARD_WRITE_GRAPH": False,

    # Size of dataset

    "SAMPLES": 10 ** 4,

    # Multiprocessing
    # "PROCESSOR_NUM": None,  # defaults to all processors,

    "MODEL_DIR": "gs://ng-training-data",  # this should go to train_policy

    'VARIABLE_NUM': 8,
    'CLAUSE_NUM': 80,
    'LEARNING_RATE': 0.001,
    # 'CLAUSE_SIZE': 3,  # not applicable for graph network
    # 'MIN_VARIABLE_NUM': 30,  # only needed for generation
    'LEVEL_NUMBER': 0,
    'BATCH_SIZE': 128
}


class Graph:
    def __init__(self, settings, features, labels):
        BATCH_SIZE = DEFAULT_SETTINGS["BATCH_SIZE"]
        VARIABLE_NUM = DEFAULT_SETTINGS["VARIABLE_NUM"]
        CLAUSE_NUM = DEFAULT_SETTINGS["CLAUSE_NUM"]
        self.inputs = features

        def assert_shape(a, b):
            del a
            del b

        self.sat_labels = tf.constant(1., dtype=tf.float32, shape=[BATCH_SIZE])
        print("sat labels shape at graph init", self.sat_labels.get_shape())
        self.policy_labels = labels
        print("policy labels shape at graph init", self.policy_labels.get_shape())

        batch_size = BATCH_SIZE
        variable_num = VARIABLE_NUM
        clause_num = CLAUSE_NUM
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

        with tf.variable_scope("graph_net", reuse=tf.AUTO_REUSE):
            initial_var_embedding = tf.get_variable(
                name='init_var_embedding',
                shape=[EMBEDDING_SIZE],
                initializer=tf.random_uniform_initializer(-1., 1))
            positive_literal_embeddings = tf.tile(
                tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
                [batch_size, variable_num, 1])
            negative_literal_embeddings = tf.tile(
                tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
                [batch_size, variable_num, 1])

            initial_clause_embedding = tf.get_variable(
                name='init_clause_embedding',
                shape=[1, 1, EMBEDDING_SIZE],
                initializer=tf.random_uniform_initializer(-1., 1))
            clause_embeddings = tf.tile(
                tf.reshape(initial_clause_embedding, [1, 1, EMBEDDING_SIZE]),
                [batch_size, clause_num, 1])

        for level in range(LEVEL_NUMBER+1):
            if level >= 1:
                assert_shape(positive_literal_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
                assert_shape(negative_literal_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                clause_preembeddings = tf.concat([
                    tf.divide((
                        tf.matmul(positive_connections, positive_literal_embeddings, transpose_a=True) +
                        tf.matmul(negative_connections, negative_literal_embeddings, transpose_a=True)
                    ), tf.expand_dims(variables_per_clause, -1) + 1.0),
                    clause_embeddings], axis=-1)
                assert_shape(clause_preembeddings,
                             [BATCH_SIZE, None, EMBEDDING_SIZE*2])
                last_hidden = clause_preembeddings
                for index, size in enumerate(HIDDEN_LAYERS):
                    last_hidden = tf.layers.dense(
                        last_hidden, size,
                        activation=HIDDEN_ACTIVATION, name='hidden_clause_{}'.format(index),
                        reuse=reuse)
                clause_embeddings = tf.layers.dense(
                    last_hidden, EMBEDDING_SIZE, activation=EMBED_ACTIVATION, name='clause_embeddings', reuse=reuse)
                assert_shape(clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                # clause -> var
                positive_literal_preembeddings = tf.concat([
                    tf.divide((
                            tf.matmul(positive_connections, clause_embeddings)
                    ), tf.expand_dims(clauses_per_variable, -1) + 1.0),
                    positive_literal_embeddings,
                    negative_literal_embeddings], axis=-1)
                negative_literal_preembeddings = tf.concat([
                    tf.divide((
                        tf.matmul(negative_connections, clause_embeddings)
                    ), tf.expand_dims(clauses_per_variable, -1) + 1.0),
                    negative_literal_embeddings,
                    positive_literal_embeddings], axis=-1)
                assert_shape(positive_literal_preembeddings, [BATCH_SIZE, None, EMBEDDING_SIZE * 3])
                assert_shape(negative_literal_preembeddings, [BATCH_SIZE, None, EMBEDDING_SIZE * 3])
                last_hidden_positive = positive_literal_preembeddings
                last_hidden_negative = negative_literal_preembeddings
                for index, size in enumerate(HIDDEN_LAYERS):
                    last_hidden_positive = tf.layers.dense(
                        last_hidden_positive, size,
                        activation=HIDDEN_ACTIVATION,
                        name='hidden_var_{}'.format(index),
                        reuse=reuse)
                    last_hidden_negative = tf.layers.dense(
                        last_hidden_negative, size,
                        activation=HIDDEN_ACTIVATION,
                        name='hidden_var_{}'.format(index),
                        reuse=reuse)
                positive_literal_embeddings = tf.layers.dense(
                    last_hidden_positive, EMBEDDING_SIZE, activation=EMBED_ACTIVATION, name='var_embeddings', reuse=reuse)
                negative_literal_embeddings = tf.layers.dense(
                    last_hidden_negative, EMBEDDING_SIZE, activation=EMBED_ACTIVATION, name='var_embeddings', reuse=reuse)
            assert_shape(positive_literal_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
            assert_shape(negative_literal_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

            self.positive_policy_logits = tf.layers.dense(
                positive_literal_embeddings, 1, name='policy', reuse=reuse)
            self.negative_policy_logits = tf.layers.dense(
                negative_literal_embeddings, 1, name='policy', reuse=reuse)
            self.policy_logits = tf.concat([self.positive_policy_logits, self.negative_policy_logits], axis=2)
            assert_shape(self.policy_logits, [BATCH_SIZE, None, 2])

            self.sat_logits = (tf.reduce_sum(
                tf.layers.dense(positive_literal_embeddings, 1, name='sat', reuse=reuse),
                axis=[1, 2]) + tf.reduce_sum(
                tf.layers.dense(negative_literal_embeddings, 1, name='sat', reuse=reuse),
                axis=[1, 2]))
            assert_shape(self.sat_logits, [BATCH_SIZE])

            # zero out policy for test when UNSAT
            # requires sat_labels to be provided, so needs to be a separate tensor in order
            # for inference to work
            print("batch size", batch_size)
            print("sat labels", self.sat_labels.get_shape())
            print("policy logits", self.policy_logits.get_shape())
            self.policy_logits_for_cmp = tf.reshape(self.sat_labels, [batch_size, 1, 1]) * self.policy_logits

            self.policy_loss = tf.losses.sigmoid_cross_entropy(
                self.policy_labels, self.policy_logits_for_cmp)
            self.policy_probabilities = tf.sigmoid(self.policy_logits, name='policy_prob')
            self.policy_probabilities_for_cmp = tf.sigmoid(self.policy_logits_for_cmp)

            self.sat_loss = tf.losses.sigmoid_cross_entropy(self.sat_labels, self.sat_logits)
            self.sat_probabilities = tf.sigmoid(self.sat_logits, name='sat_prob')

            # we do not want to count unsat into policy_error
            self.policy_error = tf.reduce_sum(tf.abs(
                tf.round(self.policy_probabilities_for_cmp) - self.policy_labels)) / (
                  tf.reduce_sum(self.sat_labels)) / (tf.cast(variable_num, dtype=tf.float32) * 2.0)
            self.sat_error = tf.reduce_mean(tf.abs(
                tf.round(self.sat_probabilities) - self.sat_labels))
        
            self.level_loss = SAT_LOSS_WEIGHT * self.sat_loss + POLICY_LOSS_WEIGHT * self.policy_loss
            self.loss += self.level_loss

            lvln = "_level_{}".format(level)
            # tf.summary.scalar("loss"+lvln, self.level_loss)
            # tf.summary.scalar("policy_loss"+lvln, self.policy_loss)
            # tf.summary.scalar("policy_error"+lvln, self.policy_error)
            # #tf.summary.scalar("policy_top1_error"+lvln, self.policy_top1_error)
            # tf.summary.scalar("sat_loss"+lvln, self.sat_loss)
            # tf.summary.scalar("sat_error"+lvln, self.sat_error)
            # tf.summary.scalar("sat_fraction"+lvln, tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))

        # tf.summary.scalar("loss", self.loss)
        # tf.summary.scalar("policy_loss", self.policy_loss)
        # tf.summary.scalar("policy_error", self.policy_error)
        # tf.summary.scalar("policy_top1_error", self.policy_top1_error)
        # tf.summary.scalar("sat_loss", self.sat_loss)
        # tf.summary.scalar("sat_error", self.sat_error)
        # tf.summary.scalar("sat_fraction",
        #                   tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))


def model_fn(features, labels, mode, params):
    del params

    graph = Graph(DEFAULT_SETTINGS, features=features, labels=labels)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode, loss=graph.loss,
            eval_metric_ops={'sat_error': graph.sat_error,
                             'policy_error': graph.policy_error})

    elif mode == tf.estimator.ModeKeys.TRAIN:
        loss = graph.loss
        optimizer = tf.train.AdamOptimizer(learning_rate=DEFAULT_SETTINGS["LEARNING_RATE"])

        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op
        )
    else:
        assert False


def dummy_sample():
    # Made just to fit the shape and to isolate actual data generation.

    sample_number = DEFAULT_SETTINGS["BATCH_SIZE"]
    variable_number = DEFAULT_SETTINGS["VARIABLE_NUM"]
    clause_num = DEFAULT_SETTINGS["CLAUSE_NUM"]

    features = np.asarray([[[[1, 0] for _ in range(clause_num)]
                            for _ in range(variable_number)]
                           for _ in range(sample_number)])

    policy_labels = np.asarray([[[1, 0] for _ in range(variable_number)]
                                for _ in range(sample_number)])
    sat_labels = np.asarray([True for _ in range(sample_number)])

    return features, sat_labels, policy_labels


def train_input_fn(params):
    del params

    batch_size = DEFAULT_SETTINGS["BATCH_SIZE"]

    variable_num = DEFAULT_SETTINGS["VARIABLE_NUM"]
    clause_num = DEFAULT_SETTINGS["CLAUSE_NUM"]

    def parser(serialized_example):
        return tf.parse_single_example(
            serialized_example,
            features={
                'inputs': tf.FixedLenFeature([variable_num, clause_num, 2], tf.float32),
                'sat': tf.FixedLenFeature([], tf.float32),
                'policy': tf.FixedLenFeature([variable_num, 2], tf.float32),
            })

    dataset = tf.data.TFRecordDataset(FLAGS.train_file)
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    # dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.map(lambda x:
                          # (x["inputs"],{"sat": x["sat"], "policy": x["policy"]}))
                          (x["inputs"], x["policy"]))

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.make_one_shot_iterator().get_next()
    print("shape inputs", dataset[0].get_shape())
    # print("shape sat", dataset[1]["sat"].get_shape())
    print("shape policy", dataset[1].get_shape())
    return dataset


def dummy_train_input_fn(params):
    del params

    features, sat_labels, policy_labels = dummy_sample()


    features, labels = (
        features.astype(np.float32),
        # {"sat": np.asarray(sat_labels).astype(np.float32),
        # "policy":
        policy_labels.astype(np.float32)#})
    )
    # return features, labels
    ds = tf.data.Dataset.from_tensors((features, labels)).repeat()
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_tpu:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu,
      )
  else:
      tpu_cluster_resolver = None

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=DEFAULT_SETTINGS["BATCH_SIZE"],
      eval_batch_size=DEFAULT_SETTINGS["BATCH_SIZE"],
      predict_batch_size=DEFAULT_SETTINGS["BATCH_SIZE"],
      config=run_config)
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=dummy_train_input_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.



if __name__ == "__main__":
  tf.app.run()
