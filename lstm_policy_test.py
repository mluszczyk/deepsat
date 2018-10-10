import tensorflow as tf

import lstm_policy


class TestGraphPolicy(tf.test.TestCase):
    @tf.test.mock.patch.dict(lstm_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                            "SAMPLES": 64})
    def test_train_not_sr(self):
        self.assertFalse(lstm_policy.DEFAULT_SETTINGS["SR_GENERATOR"])
        lstm_policy.main()

    @tf.test.mock.patch.dict(lstm_policy.DEFAULT_SETTINGS, {"NEPTUNE_ENABLED": False,
                                                             "SAMPLES": 64,
                                                             "SR_GENERATOR": True})
    def test_train_sr(self):
        lstm_policy.main()
