import os 
import tensorflow as tf 
import pickle 
import unittest
import numpy as np 
tf.app.flags.DEFINE_string("path_1",None,"Path to prediciton results of model 1")
tf.app.flags.DEFINE_string("path_2",None,"Path to prediction results of model 2")
tf.app.flags.DEFINE_string("path_3",None,"Path to prediction results of model 3")
tf.app.flags.DEFINE_string("result_path",".", "Path to results of ensemble model")
FLAGS = tf.app.flags.FLAGS

def main(_):
    paths = [FLAGS.path_1, FLAGS.path_2,FLAGS.path_3]
    predicitons = []
    labels = [] 
    names = []
    for path in paths:
        with open(path, "rb") as f :
            x = pickle.load(f)
            predicitons.append(x[0])
            labels.append(x[1])
            names.append(x[2])
    assert names[0] == names[1]
    assert names[1] == names[2]
    ensemble_results = []
    for i in range(0, len(predicitons[0])):
        vote = [0,0,0]
        for pred in predicitons:
            vote[pred[i]] = vote[pred[i]] + 1 
            ensemble_results.append(np.argmax(np.array(vote)))
    with open(os.path.join(FLAGS.result_path,"ensemble.pkl"),"wb") as f:
        print("Start to recode ensemble results to %s "%str(os.path.join(FLAGS.result_path,"ensemble.pkl")) )
        pickle.dump((ensemble_results,labels[0],names[0]),f)

if __name__ == "__main__":
    tf.app.run()            

