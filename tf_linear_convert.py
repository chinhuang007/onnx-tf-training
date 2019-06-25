import numpy as np
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf
import shutil

from tensorflow.python.platform import app
from tensorflow.python.summary import summary

def import_to_tensorboard(model_dir, log_dir):

  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], model_dir)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir {}".format(log_dir))

z = 0.01
x = [[1], [2]]
y = [[2], [4]]

# read from onnx pb to gather the model graph and training info

# first load the model graph into the tf default graph
# this should be the same as the current converter
# the feasibility is verified here with temporary changes to
# the converter code
onnx_file = 'training_linear_model.onnx'
model = onnx.load(onnx_file)
tf_rep = prepare(model)

# This is a Tensorflow graph, converted from onnx file
g = tf_rep.graph
outputs = [tf_rep.tensor_dict[output] for output in tf_rep.outputs]
y_pred = outputs[0]

# We set our graph as the tf default graph in the thread
with g.as_default():
  # then use training info to add the rest into the tf default graph

  # create placeholder based on training info additional inputs, label
  # and additional initializers, learning_rate
  p1 = tf.placeholder(tf.float32, shape=[2, 1], name='label')
  p2 = tf.placeholder(tf.float32, shape= [], name='learning_rate')

  # next we add the loss function
  # add loss to training procedure based on the training info loss node
  loss = tf.losses.mean_squared_error(labels=p1, predictions=y_pred)

  # and add the optimizer 
  # add optimizer to training procedure based on the training info optimizer node
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=p2)
  train = optimizer.minimize(loss)

# now the training graph is complete
# we can start the session next
with tf.Session(graph=g) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  # set the input tensor for training and inference
  p = tf_rep.tensor_dict[tf_rep.inputs[0]]

  # let's do training in tf and check the loss value improving
  for i in range(100):
    _, loss_value = sess.run((train, loss), 
          feed_dict={p: x, p1: y, p2: z}) 
    print(loss_value)

  # also do inference on 6 and 8
  new_x = [[6], [8]]

  print(sess.run(y_pred, feed_dict={p: new_x}))

  model_dir = '/tmp/tf-linear'
  try:
    shutil.rmtree(model_dir)
  except:
    pass

  # lastly save the model as a saved_model
  tf.saved_model.simple_save(sess,
            model_dir,
            inputs={"input": p, "label": p1, "learning_rate": p2},
            outputs={"MatMul": y_pred})

# make the trainable model viewable in TensorBoard
import_to_tensorboard(model_dir, model_dir+'/logdir')

