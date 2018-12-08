import time

import datetime
import sys

import multiprocessing
import numpy as np
import os
import tensorflow as tf

import cnf_dataset
import neurosat_estimator
import options
from reports import register_training


def train_policy(create_graph_fn, default_settings, representation='graph'):
    settings = options.get_massive_policy()

    SUMMARY_DIR = "summaries"
    MODEL_DIR = settings["MODEL_DIR"]
    MODEL_NAME = "neuropol"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    DATESTR = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")
    SUMMARY_PREFIX = SUMMARY_DIR + "/" + MODEL_NAME + "-" + DATESTR
    THIS_MODEL_DIR = MODEL_DIR + "/" + MODEL_NAME + "-" + DATESTR
    MODEL_PREFIX = THIS_MODEL_DIR + "/model"

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

    datagen_options = {k: settings[k]
                       for k in ["VARIABLE_NUM", "SR_GENERATOR", "BATCH_SIZE", "MIN_VARIABLE_NUM",
                                 "CLAUSE_NUM", "CLAUSE_SIZE", "MIN_CLAUSE_NUM", "PROCESSOR_NUM"]}

    with cnf_dataset.PoolDatasetGenerator(datagen_options) as dataset_generator:
        def input_fn(params):
            del params

            sample_with_labels = dataset_generator.generate_batch(representation=representation)
            if representation == 'graph':
                features, labels = (
                    sample_with_labels.inputs.astype(np.float32),
                    {"sat": np.asarray(sample_with_labels.sat_labels).astype(np.float32),
                     "policy": sample_with_labels.policy_labels.astype(np.float32)})
                # return features, labels
                return tf.data.Dataset.from_tensors((features, labels)).repeat()

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
