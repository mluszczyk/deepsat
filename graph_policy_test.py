import tensorflow as tf

import graph_policy


class TestGraphPolicy(tf.test.TestCase):
    @tf.test.mock.patch.dict(graph_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                            "SAMPLES": 64})
    def test_train_not_sr(self):
        self.assertFalse(graph_policy.DEFAULT_SETTINGS["SR_GENERATOR"])
        graph_policy.main()

    @tf.test.mock.patch.dict(graph_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                             "SAMPLES": 64,
                                                             "SR_GENERATOR": True})
    def test_train_sr(self):
        graph_policy.main()
