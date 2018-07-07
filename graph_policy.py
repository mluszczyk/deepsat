# coding: utf-8
import functools
from functools import wraps
import tensorflow as tf
import numpy as np
from cnf import get_random_kcnfs
import datetime
import time
from tensorflow.core.framework import summary_pb2
import sys
import json
import multiprocessing
import os
from deepsense import neptune

# HYPER PARAMETERES ------------------------------------------

# Data properties
VARIABLE_NUM = 8
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

# ------------------------------------------

LAST_TIMED = dict()


def read_settings(str_settings):
    settings = json.loads(str_settings)
    for var_name, value in settings.items():
        old_value = globals()[var_name]
        assert type(value) is type(old_value)
        print("{} = {}  # default is {}".format(var_name, value, old_value))
        globals()[var_name] = value


def set_flags():
    for arg in sys.argv[1:]:
        key = '--params='
        if not arg.startswith(key):
            continue
        str_settings = arg[len(key):]
        read_settings(str_settings)

    environ_settings = os.environ.get('DEEPSAT_PARAMS', '{}')
    read_settings(environ_settings)


def timed(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        LAST_TIMED[func.__name__] = end - start
        return results
    return new_func


def assert_shape(matrix, shape: list):
    act_shape = matrix.get_shape().as_list()
    assert act_shape == shape, "got shape {}, expected {}".format(act_shape, shape)


class Graph:
    def __init__(self):
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


def set_and_sat(triple):
    cnf, is_satisfiable, literal = triple
    if not is_satisfiable or not abs(literal) in cnf.vars:
        return False
    cnf = cnf.set_var(literal)
    return cnf.satisfiable()


def satisfiable(cnf):
    return cnf.satisfiable()


def clauses_to_matrix(clauses, max_clause_num=None, max_variable_num=None):
    if max_clause_num is None:
        max_clause_num = CLAUSE_NUM
    if max_variable_num is None:
        max_variable_num = VARIABLE_NUM
    def var_in_clause_val(var, i):
        if i >= len(clauses):
            return [0.0, 0.0]
        return [1.0 if var in clauses[i] else 0.0,
                1.0 if -var in clauses[i] else 0.0]
    result = [[var_in_clause_val(var, i) for i in range(max_clause_num)]
              for var in range(1, max_variable_num+1)]
    return result


@timed
def gen_labels(pool, cnfs):
    sat_labels = pool.map(satisfiable, cnfs)

    sats_to_check = [(cnf, is_satisfiable, literal)
                     for (cnf, is_satisfiable) in zip(cnfs, sat_labels)
                     for v in range(1, VARIABLE_NUM + 1)
                     for literal in [v, -v]]
    policy_labels = np.asarray(pool.map(set_and_sat, sats_to_check))
    policy_labels = np.asarray(policy_labels).reshape(len(cnfs), VARIABLE_NUM, 2)
    assert len(cnfs) == len(sat_labels) == policy_labels.shape[0]
    assert policy_labels.shape[1] == VARIABLE_NUM
    assert policy_labels.shape[2] == 2

    return sat_labels, policy_labels


@timed
def gen_cnfs_with_labels(pool):
    cnfs = get_random_kcnfs(BATCH_SIZE, CLAUSE_SIZE, VARIABLE_NUM, CLAUSE_NUM,
                            min_clause_number=MIN_CLAUSE_NUM)
    sat_labels, policy_labels = gen_labels(pool, cnfs)
    return cnfs, sat_labels, policy_labels


def main():
    if NEPTUNE_ENABLED:
        context = neptune.Context()
        context.integrate_with_tensorflow()


    set_flags()

    print("cpu number:", multiprocessing.cpu_count())

    tf.reset_default_graph()
    model = Graph()

    print()
    print("PARAMETERS")
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print("TOTAL PARAMS:", total_parameters)

    np.set_printoptions(precision=2, suppress=True)

    merged_summaries = tf.summary.merge_all()

    SUMMARY_DIR = "summaries"
    MODEL_DIR = "models"
    MODEL_NAME = "activepolicy"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    DATESTR = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")
    SUMMARY_PREFIX = SUMMARY_DIR + "/" + MODEL_NAME + "-" + DATESTR
    MODEL_PREFIX = MODEL_DIR + "/" + MODEL_NAME + "-" + DATESTR + "/model"
    train_writer = tf.summary.FileWriter(SUMMARY_PREFIX + "-train")

    with open(__file__, "r") as fil:
        # ending tag is broken, because we print ourselves!
        run_with = "# Program was run via:\n# {}".format(" ".join(sys.argv))
        value = "<pre>\n" + run_with + "\n" + fil.read() + "\n<" + "/pre>"
    text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary = tf.Summary()
    summary.value.add(tag="code", metadata=meta, tensor=text_tensor)
    train_writer.add_summary(summary)

    with tf.Session() as sess, multiprocessing.Pool(PROCESSOR_NUM) as pool:
        if BOARD_WRITE_GRAPH:
            train_writer.add_graph(sess.graph)

        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.loss)
        sess.run(tf.global_variables_initializer())

        @timed
        def nn_train(cnfs, sat_labels, policy_labels):
            inputs = np.asarray([clauses_to_matrix(cnf.clauses) for cnf in cnfs])
            summary, _, loss, probs = sess.run(
                [merged_summaries, train_op, model.loss,
                 model.policy_probabilities], feed_dict={
                    model.inputs: inputs,
                    model.policy_labels: policy_labels,
                    model.sat_labels: sat_labels,
                })
            train_writer.add_summary(summary, global_samples)

        @timed
        def complete_step():
            cnfs, sat_labels, policy_labels = gen_cnfs_with_labels(pool)
            nn_train(cnfs, sat_labels, policy_labels)

        saver = tf.train.Saver()

        global_samples = 0
        start_time = time.time()
        print_step = 1
        print_step_multiply = 2
        steps_number = int(SAMPLES/BATCH_SIZE) + 1
        for global_batch in range(steps_number):
            if global_batch % int(steps_number / 1000) == 0 or global_batch == print_step:
                if global_batch == print_step:
                    print_step *= print_step_multiply
                saver.save(sess, MODEL_PREFIX, global_step=global_samples)
                now_time = time.time()
                time_elapsed = now_time - start_time
                if global_batch == 0:
                    time_remaining = "unknown"
                    time_total = "unknown"
                else:
                    time_remaining = (time_elapsed / global_batch)\
                                     * (steps_number - global_batch)
                    time_total = time_remaining + time_elapsed
                print("Step {}, {}%\n"
                      "\tsteps left: {}\n"
                      "\ttime: {} s\n"
                      "\test remaining: {} s\n"
                      "\test total: {} s".format(
                    global_batch, round(100.*global_batch/steps_number, 1),
                    steps_number-global_batch, time_elapsed, time_remaining,
                    time_total))

            complete_step()

            summary_values = [
                summary_pb2.Summary.Value(tag="time_per_example_" + fun_name,
                                          simple_value=fun_time/BATCH_SIZE)
                for fun_name, fun_time in LAST_TIMED.items()
            ]
            summary = summary_pb2.Summary(value=summary_values)
            train_writer.add_summary(summary, global_samples)

            global_samples += BATCH_SIZE
        saver.save(sess, MODEL_PREFIX, global_step=global_samples)


if __name__ == "__main__":
    main()
