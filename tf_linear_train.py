import tensorflow as tf

model_dir = '/tmp/tf-linear'
def load_and_train(model_dir, to_save):

  # load the saved_model
  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], model_dir)

    init = tf.global_variables_initializer()
    sess.run(init)
    g=sess.graph

    # get handles on input and output tensors
    p = g.get_tensor_by_name('input:0')
    p1 = g.get_tensor_by_name('label:0')
    p2 = g.get_tensor_by_name('learning_rate:0')
    y_pred = g.get_tensor_by_name('MatMul:0')

    x = [[3], [4]]
    y = [[6], [8]]
    z = 0.01

    # let's do some training her to see the loss value improving
    for i in range(100):
      _, loss_value = sess.run((g.get_operation_by_name('GradientDescent'), g.get_tensor_by_name('mean_squared_error/value:0')),
              {p: x, p1: y, p2: z})
      print(loss_value)

    # also do inference on 6 and 8
    new_x = [[6], [8]]
    print(sess.run(y_pred, {p: new_x}))

    # again save the updated model as a saved_model
    import shutil
    if to_save:
      shutil.rmtree(model_dir)

      tf.saved_model.simple_save(sess,
            model_dir,
            inputs={"input": p, "label": p1, "learning_rate": p2},
            outputs={"MatMul": y_pred})
      print('model file is saved')

load_and_train(model_dir, True)
