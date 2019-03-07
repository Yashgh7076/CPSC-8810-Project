import tensorflow as tf

tf.reset_default_graph()
 
sess=tf.Session()    
#First let's load meta graph and restore weights
imported_graph = tf.train.import_meta_graph('C:/model.meta')
#saver = tf.train.import_meta_graph('C:/model.meta')
#saver.restore(sess, "C:/model")

for tensor in tf.get_default_graph().get_operations():
    print (tensor.name)