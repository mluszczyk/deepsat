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

# HYPER PARAMETERES ------------------------------------------

# Data properties
VARIABLE_NUM = 4
CLAUSE_SIZE = 3
CLAUSE_NUM = 20
MIN_CLAUSE_NUM = 1

# Neural net
EMBEDDING_SIZE = 64
LSTM_STATE_SIZE = 64
LSTM_LAYERS = 1
LEVEL_NUMBER = 2

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

# ------------------------------------------

LAST_TIMED = dict()


def set_flags():
    for arg in sys.argv[1:]:
        var_name, value = arg.split('=')
        value = json.loads(value)
        old_value = globals()[var_name]
        assert type(value) is type(old_value)
        print("{} = {}  # default is {}".format(var_name, value, old_value))
        globals()[var_name] = value


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
        self.inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, VARIABLE_NUM, None), name='inputs')
        self.policy_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, VARIABLE_NUM, 2), name='policy_labels')
        self.sat_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE,), name='sat_labels')

        positive_connections = tf.maximum(self.inputs, [0.0])
        negative_connections = tf.minimum(self.inputs, [0.0]) * -1.0
        assert_shape(positive_connections, [BATCH_SIZE, VARIABLE_NUM, None])
        assert_shape(negative_connections, [BATCH_SIZE, VARIABLE_NUM, None])

        reuse = tf.AUTO_REUSE

        self.loss = 0.0

        for level in range(LEVEL_NUMBER+1):
            if level == 0:
                initial_var_embedding = tf.Variable(
                    tf.random_uniform([EMBEDDING_SIZE], -1., 1),
                    name='init_var_embedding')
                var_embeddings = tf.tile(
                    tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
                    [BATCH_SIZE, VARIABLE_NUM, 1])
            elif level >= 1:
                positive_var_embeddings = tf.layers.dense(
                    var_embeddings, EMBEDDING_SIZE, name='positive_var', reuse=reuse)
                negative_var_embeddings = tf.layers.dense(
                    var_embeddings, EMBEDDING_SIZE, name='negative_var', reuse=reuse)
                assert_shape(positive_var_embeddings, [BATCH_SIZE, VARIABLE_NUM, EMBEDDING_SIZE])
                assert_shape(negative_var_embeddings, [BATCH_SIZE, VARIABLE_NUM, EMBEDDING_SIZE])

                clause_preembeddings = (
                    tf.matmul(positive_connections, positive_var_embeddings, transpose_a=True) +
                    tf.matmul(negative_connections, negative_var_embeddings, transpose_a=True)
                )
                clause_embeddings = tf.layers.dense(
                    clause_preembeddings, EMBEDDING_SIZE, name='clause_embeddings', reuse=reuse)
                assert_shape(clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                # clause -> var

                positive_clause_embeddings = tf.layers.dense(
                    clause_embeddings, EMBEDDING_SIZE, name='positive_clause', reuse=reuse)
                negative_clause_embeddings = tf.layers.dense(
                    clause_embeddings, EMBEDDING_SIZE, name='negative_clause', reuse=reuse)
                assert_shape(positive_clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])
                assert_shape(negative_clause_embeddings, [BATCH_SIZE, None, EMBEDDING_SIZE])

                var_preembeddings = (
                        tf.matmul(positive_connections, positive_clause_embeddings) +
                        tf.matmul(negative_connections, negative_clause_embeddings)
                )
                var_embeddings = tf.layers.dense(
                    var_preembeddings, EMBEDDING_SIZE, name='var_embeddings', reuse=reuse)
            assert_shape(var_embeddings, [BATCH_SIZE, VARIABLE_NUM, EMBEDDING_SIZE])

            self.policy_logits = tf.layers.dense(
                var_embeddings, 2, name='policy', reuse=reuse)
            assert_shape(self.policy_logits, [BATCH_SIZE, VARIABLE_NUM, 2])

            self.sat_logits = tf.reduce_sum(
                tf.layers.dense(var_embeddings, 1, name='sat', reuse=reuse),
                axis=[1, 2])
            assert_shape(self.sat_logits, [BATCH_SIZE])

            # zero out policy for test when UNSAT
            # requires sat_labels to be provided, so needs to be a separate tensor in order
            # for inference to work
            self.policy_logits_for_cmp = tf.reshape(self.sat_labels, [BATCH_SIZE, 1, 1]) * self.policy_logits

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
                  tf.reduce_sum(self.sat_labels)) / (VARIABLE_NUM * 2)
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
            tf.summary.scalar("sat_fraction"+lvln, tf.reduce_sum(self.sat_labels) / BATCH_SIZE)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("policy_loss", self.policy_loss)
        tf.summary.scalar("policy_error", self.policy_error)
        # tf.summary.scalar("policy_top1_error", self.policy_top1_error)
        tf.summary.scalar("sat_loss", self.sat_loss)
        tf.summary.scalar("sat_error", self.sat_error)
        tf.summary.scalar("sat_fraction",
                          tf.reduce_sum(self.sat_labels) / BATCH_SIZE)


def set_and_sat(triple):
    cnf, is_satisfiable, literal = triple
    if not is_satisfiable or not abs(literal) in cnf.vars:
        return False
    cnf = cnf.set_var(literal)
    return cnf.satisfiable()


def satisfiable(cnf):
    return cnf.satisfiable()


def clauses_to_matrix(clauses):
    def var_in_clause_val(var, i):
        if len(clauses) > i:
            if var in clauses[i]:
                return 1.0
            if -var in clauses[i]:
                return -1.0
        return 0.0
    result = [[var_in_clause_val(var, i) for i in range(CLAUSE_NUM)]
              for var in range(1, VARIABLE_NUM+1)]
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
        print_step_multiply = 10
        steps_number = int(SAMPLES/BATCH_SIZE) + 1
        for global_batch in range(steps_number):
            if global_batch % int(steps_number / 10) == 0 or global_batch == print_step:
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
