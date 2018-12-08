import tensorflow as tf


def model_fn_with_settings(graph_class, settings):
    def model_fn(features, labels, mode, params):
        del params

        graph = graph_class(settings, features=features, labels=labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode, loss=graph.loss,
                eval_metric_ops={'sat_error': graph.sat_error,
                                 'policy_error': graph.policy_error})

        elif mode == tf.estimator.ModeKeys.TRAIN:
            loss = graph.loss
            optimizer = tf.train.AdamOptimizer(learning_rate=settings["LEARNING_RATE"])

            if settings.get("USE_TPU"):
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )
        else:
            assert False

    return model_fn


