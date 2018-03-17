# coding: utf-8
from functools import wraps

import tensorflow as tf
import numpy as np

from cnf import get_random_kcnfs


import datetime
import time

from tensorflow.core.framework import summary_pb2

LAST_TIMED = dict()


def timed(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        LAST_TIMED[func.__name__] = end - start
        return results
    return new_func


def pad_and_concat(sequences):  # sequences shape: [batch_size, len, dims...] -> ([batch_size, maxlen, dims...], [len])
    arrays = [np.asarray(seq) for seq in sequences]
    lengths = np.asarray([array.shape[0] for array in arrays], dtype=np.int32)
    maxlen = np.max(lengths)
    arrays = [np.pad(array, [(0, maxlen - array.shape[0]), (0, 0)], 'constant', constant_values=0) for array in arrays]
    return np.asarray(arrays), lengths
    


# In[4]:


VARIABLE_NUM = 4
EMBEDDING_SIZE = 8
CLAUSE_SIZE = 3
CLAUSE_NUM = 30
MIN_CLAUSE_NUM = 1
LSTM_STATE_SIZE = 8

POLICY_LOSS_WEIGHT = 1
SAT_LOSS_WEIGHT = 1
BATCH_SIZE = 64

SAMPLES = 10 ** 6
STEPS = int(SAMPLES/BATCH_SIZE) + 1


# In[5]:


def assert_shape(matrix, shape: list):
    act_shape = matrix.get_shape().as_list()
    assert act_shape == shape, "got shape {}, expected {}".format(act_shape, shape)


# In[6]:


class Graph:
    def __init__(self):
        self.inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, None, CLAUSE_SIZE), name='inputs')
        self.lengths = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='lengths')
        self.policy_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, VARIABLE_NUM * 2), name='policy_labels')
        self.sat_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE,), name='sat_labels')
        
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
        
        lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_STATE_SIZE)
        hidden_state = tf.zeros([BATCH_SIZE, LSTM_STATE_SIZE])
        current_state = tf.zeros([BATCH_SIZE, LSTM_STATE_SIZE])
        state = hidden_state, current_state
        
        _, lstm_final_state = tf.nn.dynamic_rnn(lstm, clause_embeddings, dtype=tf.float32, 
                                               sequence_length=self.lengths
                                               )
        formula_embedding = lstm_final_state.h
            
        assert_shape(formula_embedding, [BATCH_SIZE, LSTM_STATE_SIZE])
            
        softmax_w = tf.Variable(tf.random_normal([LSTM_STATE_SIZE, VARIABLE_NUM*2]), name='softmax_w')
        softmax_b = tf.Variable(tf.random_normal([VARIABLE_NUM*2]), name='softmax_b')
        
        sat_w = tf.Variable(tf.random_normal([LSTM_STATE_SIZE, 1]), name='sat_w')
        sat_b = tf.Variable(tf.random_normal([1]), 'sat_b')
        
        self.policy_logits = formula_embedding @ softmax_w + softmax_b
        
        self.sat_logits = tf.squeeze(formula_embedding @ (sat_w), axis=1) + sat_b
        
        # zero out policy for test when UNSAT
        # requires sat_labels to be provided, so needs to be a separate tensor in order 
        # for inference to work
        self.policy_logits_for_cmp = tf.expand_dims(self.sat_labels, axis=1) * self.policy_logits
        
        self.policy_loss = tf.losses.sigmoid_cross_entropy(
            self.policy_labels, self.policy_logits_for_cmp) 
        self.policy_probabilities = tf.sigmoid(self.policy_logits)
        self.policy_probabilities_for_cmp = tf.sigmoid(self.policy_logits_for_cmp)
        
        self.sat_loss = tf.losses.sigmoid_cross_entropy(self.sat_labels, self.sat_logits)
        self.sat_probabilities = tf.sigmoid(self.sat_logits)

        self.policy_top1_error = 1.0 - tf.reduce_sum(tf.gather_nd(
            self.policy_labels,
            tf.stack([tf.range(BATCH_SIZE), tf.argmax(self.policy_probabilities_for_cmp, axis=1, output_type=tf.int32)],
                     axis=1))) / (tf.reduce_sum(self.sat_labels))

        self.policy_error = tf.reduce_sum(tf.abs(
            tf.round(self.policy_probabilities_for_cmp) - self.policy_labels)) / (
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

tf.reset_default_graph()
model = Graph()


# In[20]:


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


# In[7]:


np.set_printoptions(precision=2, suppress=True)


@timed
def gen_labels(cnfs):
    sat_labels = [cnf.satisfiable() for cnf in cnfs]

    policy_labels = []
    for cnf, is_satisfiable in zip(cnfs, sat_labels):
        if is_satisfiable:
            correct_steps = cnf.get_correct_steps()
            label = []
            # for every variable, for true, then for false
            for v in range(1, VARIABLE_NUM+1):
                for sv in [v, -v]:
                    result = 1.0 if sv in correct_steps else 0.0
                    label.append(result)
            policy_labels.append(label)
        else:
            policy_labels.append([0 for _ in range(VARIABLE_NUM * 2)])

    assert all(len(label) == 2 * VARIABLE_NUM for label in policy_labels)
    sat_steps = sum(sum(label) for label in policy_labels)
    unsat_steps = len(cnfs) * 2 * VARIABLE_NUM - sat_steps
    assert unsat_steps * 4 > sat_steps
    assert len(cnfs) == len(sat_labels) == len(policy_labels)

    return sat_labels, policy_labels


@timed
def gen_cnfs_with_labels():
    cnfs = get_random_kcnfs(BATCH_SIZE, CLAUSE_SIZE, VARIABLE_NUM, CLAUSE_NUM,
                            min_clause_number=MIN_CLAUSE_NUM)
    sat_labels, policy_labels = gen_labels(cnfs)
    return cnfs, sat_labels, policy_labels


merged_summaries = tf.summary.merge_all()

SUMMARY_DIR = "summaries"
MODEL_NAME = "activepolicy"
DATESTR = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")
SUMMARY_PREFIX = SUMMARY_DIR + "/" + MODEL_NAME + "-" + DATESTR
train_writer = tf.summary.FileWriter(SUMMARY_PREFIX + "-train")
test_writer = tf.summary.FileWriter(SUMMARY_PREFIX + "-test")


with tf.Session() as sess:
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(model.loss)
    sess.run(tf.global_variables_initializer())

    @timed
    def nn_train(cnfs, sat_labels, policy_labels):
        inputs, lengths = pad_and_concat([cnf.clauses for cnf in cnfs])
        summary, _, loss, probs = sess.run(
            [merged_summaries, train_op, model.loss,
             model.policy_probabilities], feed_dict={
                model.inputs: inputs,
                model.policy_labels: policy_labels,
                model.lengths: lengths,
                model.sat_labels: sat_labels
            })
        train_writer.add_summary(summary, global_samples)

    @timed
    def complete_step():
        cnfs, sat_labels, policy_labels = gen_cnfs_with_labels()
        nn_train(cnfs, sat_labels, policy_labels)

    global_samples = 0
    start_time = time.time()
    for global_batch in range(STEPS):
        if global_batch % int(STEPS / 10) == 0 or global_batch < 100:
            now_time = time.time()
            time_elapsed = now_time - start_time
            if global_batch == 0:
                time_remaining = "unknown"
                time_total = "unknown"
            else:
                time_remaining = (time_elapsed / global_batch)\
                                 * (STEPS - global_batch)
                time_total = time_remaining + time_elapsed
            print("Step {}\n"
                  "\tsteps left: {}\n"
                  "\ttime: {} s\n"
                  "\test remaining: {} s\n"
                  "\test total: {} s".format(
                global_batch, STEPS-global_batch, time_elapsed, time_remaining,
                time_total))

        complete_step()

        summary_values = [
            summary_pb2.Summary.Value(tag="time_per_example" + fun_name,
                                      simple_value=fun_time/BATCH_SIZE)
            for fun_name, fun_time in LAST_TIMED.items()
        ]
        summary = summary_pb2.Summary(value=summary_values)
        train_writer.add_summary(summary, global_samples)

        global_samples += BATCH_SIZE
