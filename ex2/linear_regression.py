import tensorflow as tf

def linear_regression(X, y, mode, params):

    W = tf.get_variable("W", [X.shape[1], 1])
    b = tf.get_variable("b", [1])

    pred_y = tf.add(b, tf.matmul(X, W))
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "predicted_y": pred_y,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.mean_squared_error(y, pred_y)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training operation
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.compat.v2.optimizers.SGD(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):