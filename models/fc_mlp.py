import os
import numpy as np
import tensorflow as tf
from models.base_model import Model
from tensorflow.contrib.tensorboard.plugins import projector

class FC_MLP(Model):
    def __init__(self, params):
        super(FC_MLP, self).__init__(params)

    def load_params(self, params):
        # TODO: type controlling
        super(FC_MLP, self).load_params(params)
        # Training Schedule
        self.weight_lr = np.float32(params.weight_learning_rate)
        self.lr_staircase = params.lr_staircase
        self.lr_decay_rate = np.float32(params.lr_decay_rate)
        self.lr_decay_steps = np.float32(params.lr_decay_steps)
        self.weight_decay_mults = [np.float32(mult) for mult in params.weight_decay_mults]
        self.orth_mults = [np.float32(mult) for mult in params.orth_mults]
        # Regularizers
        self.do_batch_norm = params.do_batch_norm
        if self.do_batch_norm:
            self.norm_decay_mult = np.float32(params.norm_decay_mult)
        # Architecture
        self.data_shape = params.data_shape
        self.batch_size = params.batch_size
        self.output_channels = params.fc_output_channels
        self.optimizer_type = params.optimizer_type
        # Calculated params
        self.x_shape = [None,] + self.data_shape
        self.y_shape = [None, 2] # control or patient one-hot (column 0 marks control, column 1 marks patient)
        self.b_shapes = [val for val in self.output_channels]
        input_channels = [self.data_shape[-1]] + self.output_channels[:-1]
        self.w_shapes = [vals for vals in zip(input_channels, self.output_channels)]
        self.num_layers = len(self.output_channels)

    def compute_orthogonalization_loss(self, multipliers, weights):
        """
        Implements weight orthogonalization loss, which encourages weights to be more orthogonal
        Parameters:
            multipliers: list of tf placeholders that contain the orthogonalization multipliers
            weights: tensor of shape [num_inputs, num_outputs]
        Outputs:
            loss: mean(m_i * (abs(W^T * W) - I))
        """
        w_orth_list = [loss_mult * tf.reduce_sum(tf.abs(tf.subtract(tf.matmul(tf.transpose(weight),
            weight), tf.eye(num_rows=weight_shape[1]))))
            for loss_mult, weight, weight_shape in zip(multipliers, weights, self.w_shapes)]
        orthogonalization_loss = tf.add_n(w_orth_list, name="orthogonalization_loss")
        return orthogonalization_loss

    def compute_weight_decay_loss(self, multipliers, weights):
        """
        Implements weight decay loss, which equals l2 regularization for SGD
        Parameters:
            multipliers: list of tf placeholders that contain the decay multipliers
            weights: tensor of shape [num_inputs, num_outpus].
        Outpus:
            loss: sum_i { m_i/2 * ||w_i||_2^2 }
        """
        indiv_loss = [tf.multiply(np.float32(multipliers[w_idx]/2.0),
            tf.reduce_sum(tf.square(weight)))
            for w_idx, weight in enumerate(weights)]
        loss = tf.add_n(indiv_loss, name="weight_decay_loss")
        return loss

    def batch_normalization(self, layer_id, a_in, reduc_axes=[0]):
        """
        Implements batch normalization
        Parameters:
            layer_id [int] index for layer
            a_in [tensor] feature map (or vector) for layer
            reduc_axes [list of ints] what axes to reduce over; default is for fc layer,
                [0,1,2] should be used for a conv layer
        #TODO: for conv, this computes batch norm over batch, x, and y dimensions
        #    it might also be useful to compute it only over batch dimensions
        """
        input_mean, input_var = tf.nn.moments(a_in, axes=reduc_axes)
        self.layer_means[layer_id] = ((1 - self.norm_decay_mult) * self.layer_means[layer_id]
            + self.norm_decay_mult * input_mean)
        self.layer_vars[layer_id] = ((1 - self.norm_decay_mult) * self.layer_vars[layer_id]
            + self.norm_decay_mult * input_var)
        adj_a_in = tf.divide(tf.subtract(a_in, self.layer_means[layer_id]),
            tf.sqrt(tf.add(self.layer_vars[layer_id], self.eps)))
        act_out = tf.add(tf.multiply(self.batch_norm_scale[layer_id], adj_a_in),
          self.batch_norm_shift[layer_id])
        return act_out

    def make_layers(self):
        a_list = [self.x]
        w_list = []
        b_list = []
        for layer_id in range(self.num_layers):
            a_out, w, b = self.fc_layer_maker(layer_id, a_list[layer_id],
                self.w_shapes[layer_id], self.b_shapes[layer_id])
            a_list.append(a_out)
            w_list.append(w)
            b_list.append(b)
        return a_list, w_list, b_list

    def fc_layer_maker(self, layer_id, a_in, w_shape, b_shape):
        w_init = tf.truncated_normal_initializer(stddev=1/w_shape[0], dtype=tf.float32)
        with tf.variable_scope(self.weight_scope) as scope:
            w = tf.get_variable(name="w"+str(layer_id), shape=w_shape, dtype=tf.float32,
                initializer=w_init, trainable=True)
            b = tf.get_variable(name="b"+str(layer_id), shape=b_shape, dtype=tf.float32,
                initializer=self.b_init, trainable=True)
        with tf.variable_scope("layer"+str(layer_id)) as scope:
            fc_out = tf.nn.relu(tf.add(tf.matmul(a_in, w), b), name="fc_out"+str(layer_id))
            if self.do_batch_norm:
                fc_out = self.batch_normalization(layer_id, fc_out, reduc_axes=[0])
        return fc_out, w, b

    def setup_graph(self):
        with tf.device(self.device):
            with self.graph.as_default():
                with tf.name_scope("placeholders") as scope:
                    self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
                    self.y = tf.placeholder(tf.float32, shape=self.y_shape, name="input_label")

                with tf.name_scope("step_counter") as scope:
                    self.global_step = tf.Variable(0, trainable=False, name="global_step")

                with tf.name_scope("weight_inits") as scope:
                    self.w_init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
                    self.b_init = tf.initializers.zeros(dtype=tf.float32)

                with tf.variable_scope("weights") as scope:
                    self.weight_scope = tf.get_variable_scope()
                    if self.do_batch_norm:
                        self.batch_norm_scale = [tf.get_variable(name="batch_norm_scale"+str(idx),
                            dtype=tf.float32, initializer=tf.constant(1.0)) for idx in
                            range(self.num_layers)]
                        self.batch_norm_shift = [tf.get_variable(name="batch_norm_shift"+str(idx),
                            dtype=tf.float32, initializer=tf.constant(0.0)) for idx in
                            range(self.num_layers)]
                        self.layer_means = [tf.Variable(tf.zeros([num_layer_features]),
                            dtype=tf.float32, trainable=False)
                            for num_layer_features in self.output_channels]
                        self.layer_vars = [tf.Variable(0.01*tf.ones([num_layer_features]),
                            dtype=tf.float32, trainable=False)
                            for num_layer_features in self.output_channels]

                self.a_list, self.w_list, self.b_list = self.make_layers()

                with tf.name_scope("prediction") as scope:
                    self.y_ = tf.nn.softmax(self.a_list[-1])

                with tf.name_scope("loss") as scope:
                    self.cross_entropy_loss = -tf.reduce_mean(tf.multiply(self.y,
                        tf.log(tf.clip_by_value(self.y_, self.eps, 1.0))),
                        name="cross_entropy_loss")
                    self.decay_loss = self.compute_weight_decay_loss(self.weight_decay_mults, self.w_list)
                    self.orth_loss = self.compute_orthogonalization_loss(self.orth_mults, self.w_list)
                    self.total_loss = tf.add_n([self.cross_entropy_loss, self.orth_loss, self.decay_loss],
                        name="total_loss")

                with tf.name_scope("optimizer") as scope:
                    #TODO: Different learning rate for different layers
                    learning_rates = tf.train.exponential_decay(
                        learning_rate = self.weight_lr,
                        global_step = self.global_step,
                        decay_steps = self.lr_decay_steps,
                        decay_rate = self.lr_decay_rate,
                        staircase = self.lr_staircase,
                        name="annealing_schedule")
                    if self.optimizer_type == "sgd":
                        self.optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                            name="sgd_optimizer")
                    elif self.optimizer_type == "adam":
                        self.optimizer = tf.train.AdamOptimizer(learning_rates, beta1=0.9,
                             beta2=0.99, epsilon=1e-07, name="adam_optimizer")
                    else:
                        assert False, ("optimizer_type parameter must be 'sgd' or 'adam'")
                    train_var_list = self.w_list + self.b_list
                    if self.do_batch_norm:
                        train_var_list += self.batch_norm_scale
                        train_var_list += self.batch_norm_shift
                    self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss,
                        var_list = train_var_list)
                    self.apply_grads = self.optimizer.apply_gradients(self.grads_and_vars,
                        global_step=self.global_step, name="model_minimizer")
