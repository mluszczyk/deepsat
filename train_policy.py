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
import neurosat_estimator
from reports import register_training
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

    SUMMARY_DIR = "summaries"
    MODEL_DIR = "models"
    MODEL_NAME = "neuropol"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    DATESTR = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")
    SUMMARY_PREFIX = SUMMARY_DIR + "/" + MODEL_NAME + "-" + DATESTR
    THIS_MODEL_DIR = MODEL_DIR + "/" + MODEL_NAME + "-" + DATESTR
    MODEL_PREFIX = THIS_MODEL_DIR + "/model"

    if settings["NEPTUNE_ENABLED"]:
        context = neptune.Context()
        context.integrate_with_tensorflow()
    else:
        context = None
    register_training.register_training(checkpoint_dir=THIS_MODEL_DIR,
                                        neptune_context=context,
                                        settings=settings,
                                        representation=representation)

    print("cpu number:", multiprocessing.cpu_count())

    tf.reset_default_graph()

    if settings["USE_TPU"]:
        master = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()
    else:
        master = None

    run_config = tf.contrib.tpu.RunConfig(
        model_dir=THIS_MODEL_DIR,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(),
        master=master)

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=neurosat_estimator.model_fn_with_settings(create_graph_fn, settings),
        model_dir=THIS_MODEL_DIR,
        config=run_config,
        train_batch_size=settings["BATCH_SIZE"],
        use_tpu=settings["USE_TPU"]
    )

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

        sess.run(tf.global_variables_initializer())

        def input_fn(params):
            del params

            sample_with_labels = dataset_generator.generate_batch(representation=representation)
            if representation == 'graph':
                return (
                    sample_with_labels.inputs.astype(np.float32),
                    {"sat": np.asarray(sample_with_labels.sat_labels).astype(np.float32),
                     "policy": sample_with_labels.policy_labels.astype(np.float32)})

            elif representation == 'sequence':
                assert False
                feed_dict = {
                    model.inputs: sample_with_labels.inputs,
                    model.lengths: sample_with_labels.lengths,
                    model.policy_labels: sample_with_labels.policy_labels,
                    model.sat_labels: sample_with_labels.sat_labels,
                }
            else:
                assert False

        estimator.train(input_fn=input_fn,
                        max_steps=max(1, settings["SAMPLES"] // settings["BATCH_SIZE"]))
