import time

import datetime
import sys

import json
import multiprocessing
import numpy as np
import os
import tensorflow as tf
from deepsense import neptune
from tensorflow.core.framework import summary_pb2

import cnf_dataset
from timed import timed, LAST_TIMED


def read_settings(str_settings_update, settings):
    settings_update = json.loads(str_settings_update)
    for var_name, value in settings_update.items():
        old_value = settings[var_name]
        assert type(value) is type(old_value)
        print("{} = {}  # default is {}".format(var_name, value, old_value))
        settings[var_name] = value


def set_flags(default_settings):
    settings = default_settings.copy()
    for arg in sys.argv[1:]:
        key = '--params='
        if not arg.startswith(key):
            continue
        str_settings = arg[len(key):]
        read_settings(str_settings, settings)

    environ_settings = os.environ.get('DEEPSAT_PARAMS', '{}')
    read_settings(environ_settings, settings)
    return settings


def train_policy(create_graph_fn, default_settings, representation='graph'):
    settings = set_flags(default_settings)

    if settings["NEPTUNE_ENABLED"]:
        context = neptune.Context()
        context.integrate_with_tensorflow()

    print("cpu number:", multiprocessing.cpu_count())

    tf.reset_default_graph()
    model = create_graph_fn(settings)

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
    MODEL_NAME = "neuropol"

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

    datagen_options = {k: settings[k]
                       for k in ["VARIABLE_NUM", "SR_GENERATOR", "BATCH_SIZE", "MIN_VARIABLE_NUM",
                                 "CLAUSE_NUM", "CLAUSE_SIZE", "MIN_CLAUSE_NUM", "PROCESSOR_NUM"]}

    with tf.Session() as sess, cnf_dataset.PoolDatasetGenerator(datagen_options) as dataset_generator:
        if settings["BOARD_WRITE_GRAPH"]:
            train_writer.add_graph(sess.graph)

        train_op = tf.train.AdamOptimizer(learning_rate=settings["LEARNING_RATE"]).minimize(model.loss)
        sess.run(tf.global_variables_initializer())

        # inputs represent green arrows, that is the "or" operation
        @timed
        def nn_train(sample_with_labels):
            if representation == 'graph':
                feed_dict = {
                    model.inputs: sample_with_labels.inputs,
                    model.policy_labels: sample_with_labels.policy_labels,
                    model.sat_labels: sample_with_labels.sat_labels,
                }
            elif representation == 'sequence':
                feed_dict = {
                    model.inputs: sample_with_labels.inputs,
                    model.lengths: sample_with_labels.lengths,
                    model.policy_labels: sample_with_labels.policy_labels,
                    model.sat_labels: sample_with_labels.sat_labels,
                }
            else:
                assert False

            summary, _, loss, probs = sess.run(
                [merged_summaries, train_op, model.loss,
                 model.policy_probabilities], feed_dict=feed_dict)

            train_writer.add_summary(summary, global_samples)

        @timed
        def complete_step():
            sample_with_labels = dataset_generator.generate_batch(representation=representation)
            nn_train(sample_with_labels)

        saver = tf.train.Saver()

        global_samples = 0
        start_time = time.time()
        print_step = 1
        print_step_multiply = 2
        steps_number = int(settings["SAMPLES"]/settings["BATCH_SIZE"]) + 1
        for global_batch in range(steps_number):
            if global_batch % max(int(steps_number / 1000), 1) == 0 or global_batch == print_step:
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

            if global_batch % 10 == 0:
                summary_values = [
                    summary_pb2.Summary.Value(tag="time_per_example_" + fun_name,
                                              simple_value=fun_time/settings["BATCH_SIZE"])
                    for fun_name, fun_time in LAST_TIMED.items()
                ]
                summary = summary_pb2.Summary(value=summary_values)
                train_writer.add_summary(summary, global_samples)

            global_samples += settings["BATCH_SIZE"]
        saver.save(sess, MODEL_PREFIX, global_step=global_samples)
