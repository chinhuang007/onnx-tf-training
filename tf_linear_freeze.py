import tensorflow as tf

sess = tf.Session()
model_dir = '/tmp/tf-linear'
tf.saved_model.loader.load(
    sess, [tf.saved_model.tag_constants.SERVING], model_dir)

def frozen():
  output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, # The session is used to retrieve the weights
    tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
    ['MatMul'] # The output node names are used to select the usefull nodes
  )

  # Finally we serialize and dump the output graph to the filesystem
  with tf.gfile.GFile(model_dir+'/linear_frozen_model.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())

frozen()
