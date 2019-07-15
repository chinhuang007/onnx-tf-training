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

z = 0.2

x = [[1.0],[2.0],[3.0],[4.0],[5.0]]
y = [[2.0],[4.0],[6.0],[8.0],[10.0]]
# read from onnx pb to gather the model graph and training info

# first load the model graph into the tf default graph
# this should be the same as the current converter
# the feasibility is verified here with temporary changes to
# the converter code
# we should also set variables as trainable based on update binding
onnx_file = 'inference_linear_model.onnx'
model = onnx.load(onnx_file)
tf_rep = prepare(model)

# since the converter doesn't work with function nodes,
# we load the training info separately
onnx_file = 'training_linear_model.onnx'
model = onnx.load(onnx_file)

# This is a Tensorflow graph, converted from onnx file
g = tf_rep.graph
outputs = [tf_rep.tensor_dict[output] for output in tf_rep.outputs]
y_pred = outputs[0]

# We set our graph as the tf default graph in the thread
with g.as_default():
  # then use training info to add the rest into the tf default graph

  # create placeholder based on training info inputs for label
  # and initializers for learning_rate
  training_inputs = model.training_info[0].input
  training_initializers = model.training_info[0].initializer

  # handle training inputs, for ex. get the label name and shape
  for n in training_inputs:
      if n.name == 'label':
        p1_name = n.name
        p1_shape = list(
          d.dim_value for d in n.type.tensor_type.shape.dim)

  # handle training initializers, for ex. set the learning rate
  for n in training_initializers:
      if n.name == 'R':
        z = n.float_data[0]
      
  p1 = tf.placeholder(tf.float32, shape=p1_shape, name=p1_name)
  p2 = tf.placeholder(tf.float32, shape= [], name='learning_rate')

  # next we add the loss function
  # add loss to training procedure based on the training info loss node
  loss = tf.losses.mean_squared_error(labels=p1, predictions=y_pred)

  # and add the optimizer 
  # add optimizer to training procedure based on the training info optimizer node
  optimizer = tf.train.AdagradOptimizer(learning_rate=p2)

  # use trainable variables for gradients
  grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
  train = optimizer.apply_gradients(grads)

# now the training graph is complete
# we can start the session next
with tf.Session(graph=g) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  # set the input tensor for training and inference
  p = tf_rep.tensor_dict[tf_rep.inputs[0]]

  # also do inference on [10],[20],[30],[40],[50]
  new_x = [[10],[20],[30],[40],[50]]

  print('Predict before training for [10, 20, 30, 40, 50] is, ', sess.run(y_pred, feed_dict={p: new_x}))

  # let's do training in tf and check the loss value improving
  for i in range(10):
    _, loss_value = sess.run((train, loss), 
          feed_dict={p: x, p1: y, p2: z}) 

  # also print the results after training
  print('Predict after training for [10, 20, 30, 40, 50] is, ', sess.run(y_pred, feed_dict={p: new_x}))

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

