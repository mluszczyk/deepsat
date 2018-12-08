import tensorflow as tf
import numpy as np

import cnf_dataset


def make_example(inputs, sat, policy):
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "inputs": tf.train.Feature(float_list=tf.train.FloatList(value=list(inputs.flatten()))),
            "sat": tf.train.Feature(
                float_list=tf.train.FloatList(value=list(sat.flatten()))),
            "policy": tf.train.Feature(
                float_list=tf.train.FloatList(value=list(policy.flatten())))
        })
    )
    return example.SerializeToString()


def tf_serialize_example(sample):
    tf_string = tf.py_func(make_example, (sample["inputs"], sample["sat"], sample["policy"]), tf.string)
    return tf.reshape(tf_string, ())


def main():
    filename = "test.tfrecord"
    options = {
        "PROCESSOR_NUM": 4,
        "CLAUSE_NUM": 80,
        "VARIABLE_NUM": 8,
        "MIN_VARIABLE_NUM": 1,
        "BATCH_SIZE": 1,
        "SR_GENERATOR": True
    }
    n_observations = 100

    with cnf_dataset.PoolDatasetGenerator(options) as generator, \
            tf.python_io.TFRecordWriter(filename) as writer:

        for _ in range(n_observations):
            sample_with_labels = generator.generate_batch()
            tf_sample = {
                 "inputs": np.squeeze(sample_with_labels.inputs.astype(np.float32), 0),
                 "sat": np.squeeze(np.asarray(sample_with_labels.sat_labels).astype(np.float32), 0),
                 "policy": np.squeeze(sample_with_labels.policy_labels.astype(np.float32), 0)}

            serialized = make_example(**tf_sample)
            writer.write(serialized)


if __name__ == '__main__':
    main()