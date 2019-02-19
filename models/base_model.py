import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

class Model(object):
    def __init__(self, params):
        self.load_params(params)
        self.make_dirs()
        self.graph = tf.Graph()
        self.set_random_seed()
        self.setup_graph()
        self.add_graph_metrics()
        self.construct_savers()
        self.add_init_ops_to_graph()

    def load_params(self, params):
        # Meta-parameters
        self.model_name = params.model_name
        self.device = params.device
        self.out_dir = params.out_dir
        self.rand_seed = params.rand_seed
        self.eps = params.eps
        self.num_epochs = params.num_epochs
        self.num_crossvalidations = params.num_crossvalidations
        self.max_cp_to_keep = params.max_cp_to_keep
        self.cp_save_dir = self.out_dir+"/"+self.model_name+"/checkpoints/"
        self.cp_latest_filename = "latest_checkpoint"
        self.weights_save_name = self.cp_save_dir + self.model_name + "_weights"
        self.full_save_name = self.cp_save_dir + self.model_name + "_full"

    def set_random_seed(self):
        with self.graph.as_default():
            tf.set_random_seed(self.rand_seed)

    def setup_graph(self):
        raise NotImplementedError

    def make_dirs(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.exists(self.cp_save_dir):
            os.makedirs(self.cp_save_dir)

    def write_graph(self, graph_def):
        write_name = self.model_name+".pb"
        tf.train.write_graph(graph_def, logdir=self.cp_save_dir, name=write_name, as_text=False)

    def add_graph_metrics(self):
        with self.graph.as_default():
            with tf.name_scope("savers") as scope:
                with tf.name_scope("performance_metrics") as scope:
                    with tf.name_scope("prediction_bools"):
                        y_max = tf.argmax(self.y_, axis=1)
                        ymax = tf.argmax(self.y, axis=1)
                        self.correct_prediction = tf.equal(y_max, ymax, name="individual_accuracy")

                    with tf.name_scope("prediction_category_stats"):
                        self.cfm = tf.confusion_matrix(ymax, y_max, num_classes=2)
                        self.true_positives = self.cfm[1,1]
                        self.true_negatives = self.cfm[0,0]
                        self.false_positives = self.cfm[0,1]
                        self.false_negatives = self.cfm[1,0]

                    with tf.name_scope("accuracy"):
                        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                            tf.float32), name="avg_accuracy")

                    with tf.name_scope("sensitivity"):
                        self.sensitivity = tf.divide(self.true_positives,
                            tf.add(self.true_positives, self.false_negatives),
                            name="avg_senitivity")

                    with tf.name_scope("specificity"):
                        self.specificity = tf.divide(self.true_negatives,
                            tf.add(self.true_negatives, self.false_positives),
                            name="avg_specificity")

                with tf.name_scope("summaries") as scope:
                    tf.summary.histogram("input", self.a_list[0])
                    [tf.summary.histogram("a"+str(idx), a) for idx, a in enumerate(self.a_list)]
                    [tf.summary.histogram("w"+str(idx), w) for idx, w in enumerate(self.w_list)]
                    [tf.summary.histogram("b"+str(idx), b) for idx, b in enumerate(self.b_list)]
                    if hasattr(self, "cross_entropy_loss"):
                        tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
                    if hasattr(self, "decay_loss"):
                        tf.summary.scalar("decay_loss", self.decay_loss)
                    tf.summary.scalar("total_loss", self.total_loss)
                    tf.summary.scalar("train_accuracy", self.accuracy)
                    tf.summary.scalar("train_sensitivity", self.sensitivity)
                    tf.summary.scalar("train_specificity", self.specificity)
                self.writer = tf.summary.FileWriter(self.cp_save_dir, graph=self.graph)
                proj_config = projector.ProjectorConfig()
                embedding = proj_config.embeddings.add()
                embedding.tensor_name = self.a_list[0].name
                projector.visualize_embeddings(self.writer, proj_config)
                self.merged_summaries = tf.summary.merge_all()

    def construct_savers(self):
        with self.graph.as_default():
            with tf.name_scope("savers") as scope:
                self.weight_saver = tf.train.Saver(var_list=self.w_list+self.b_list,
                    max_to_keep=self.max_cp_to_keep)
                self.full_saver = tf.train.Saver(max_to_keep=self.max_cp_to_keep)
        weight_saver_def = self.weight_saver.as_saver_def()
        weight_file = self.cp_save_dir+self.model_name+"_weights_saver.def"
        with open(weight_file, "wb") as f:
            f.write(weight_saver_def.SerializeToString())
            f.close()
        full_saver_def = self.full_saver.as_saver_def()
        full_file = self.cp_save_dir+self.model_name+"_full_saver.def"
        with open(full_file, "wb") as f:
            f.write(full_saver_def.SerializeToString())
            f.close()

    def add_init_ops_to_graph(self):
        with tf.device(self.device):
            with self.graph.as_default():
                with tf.name_scope("initialization") as scope:
                    self.init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    def write_checkpoint(self, session):
        weight_cp_filename = self.weight_saver.save(session,
            save_path=self.weights_save_name,
            global_step=self.global_step,
            latest_filename=self.cp_latest_filename+"_weights")
        full_cp_filename = self.full_saver.save(session,
            save_path=self.full_save_name,
            global_step=self.global_step,
            latest_filename=self.cp_latest_filename+"_full")
        return (weight_cp_filename, full_cp_filename)

    def load_weights(self, session, model_dir):
        self.weight_saver.restore(session, model_dir)
