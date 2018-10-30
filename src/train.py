import tensorflow as tf
import os
import logging
import shutil
from src.preprocessing import train_test_split, get_batch_data


class Trainner(object):
    """
    Trains a siamese RNN net instance
    """

    def __init__(self, net, batch_size=1, verification_batch_size=4, optimizer='momentum', **kwargs):
        """

        :param net: the siamese RNN net instance to train
        :param batch_size: size of training batch
        :param verification_batch_size:  size of verification batch
        :param optimizer: (optional) name of the optimizer to use (momentum or adam)
        :param kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
        """
        self.net = net
        self.batch_size = batch_size
        self.verification_batch_size = verification_batch_size
        self.optimizer_name = optimizer
        self.opt_paras = kwargs

    def _get_optimizer(self, training_iters, global_steps):

        if self.optimizer_name == 'momentum':
            learning_rate = self.opt_paras.pop('learning_rate', 0.1)
            decay_rate = self.opt_paras.pop("decay_rate", 0.95)
            momentum = self.opt_paras('momentum', 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_steps,
                                                                 decay_steps=training_iters, decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_paras).minimize(self.net.cost, global_step=global_steps)

        if self.optimizer_name == 'adam':
            learning_rate = self.opt_paras.pop('learning_rate', 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, **self.opt_paras).minimize(
                self.net.cost, global_step=global_steps)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        """

        :param training_iters: training iteration in an epoch
        :param output_path: path to save the trained model
        :param restore: boolean, True: restore previous trained model; False: restrain the model
        :param prediction_path:
        :return:
        """

        global_steps = tf.Variable(0, trainable=False, name='global_steps')

        tf.summary.scalar('cost', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)
        tf.summary.scalar('recall', self.net.recall)
        tf.summary.scalar('F1_score', self.net.F1_score)

        self.optimizer = self._get_optimizer(training_iters=training_iters, global_steps=global_steps)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        init = tf.global_variables_initializer()

        self.prediction_path = os.path.abspath(prediction_path)
        self.output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing {0}".format(self.prediction_path))
            shutil.rmtree(self.prediction_path, ignore_errors=True)
            logging.info("Removing {0}".format(self.output_path))
            shutil.rmtree(self.output_path, ignore_errors=True)

        if not os.path.exists(self.prediction_path):
            logging.info("Creating directory {0}".format(self.prediction_path))
            os.mkdir(self.prediction_path)

        if not os.path.exists(self.output_path):
            logging.info("Creating directory {0}".format(self.output_path))
            os.mkdir(self.output_path)

        return init

    def train(self, output_path, X, mask, Y, prediction_path, frac, epochs=100, restore=False):
        """

        :param output_path: path to save the model ( ckpt file etc.)
        :param X: dict={'left': sentence1_list, 'right': sentence2_list}
        :param mask: dict={'left': mask1_list, 'right': mask2_list}
        :param Y: label, list
        :param prediction_path:
        :param frac: the fraction of the data using to train the model
        :param epochs: int,
        :param restore: boolean, True: restore the previous trained model; False: retrain the model
        :return:
        """
        save_path = os.path.join(output_path, 'model.ckpt')

        if epochs == 0:
            return save_path

        # split data into train set and validation set
        train_data, validation_data = train_test_split(X, mask, Y, frac)

        n_training = len(train_data[2])
        n_val = len(validation_data[2])

        training_iters = int(n_training / self.batch_size) + 1

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.Session() as sess:
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            #train_acc=[]
            val_acc=[]
            for epoch in range(epochs):
                total_loss = 0
                for iter in range(training_iters):
                    batch_X, batch_mask, batch_Y = get_batch_data(train_data, iter * self.batch_size, self.batch_size)
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                           feed_dict={self.net.x1: batch_X['left'],
                                                      self.net.x2: batch_X['right'],
                                                      self.net.mask1: batch_mask['left'],
                                                      self.net.mask2: batch_mask['right'],
                                                      self.net.y: batch_mask[Y]})

                    total_loss += loss
                    self.output_minibatch_stats(sess, summary_writer, iter, batch_X, batch_mask, batch_Y)
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                # Evaluate the model using validation data
                val_X, val_mask, val_Y = get_batch_data(validation_data, 0, n_val)
                _, acc=self.store_prediction(sess, val_X, val_mask, val_Y, epoch)
                val_acc[epoch]=acc
                save_path = self.net.save(sess, save_path)
                # Monitor the validation accuracy
                #if self.stop_early(10, val_acc):
                 #   break
            logging.info("Optimization finished !!!")
            return save_path

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, iter, batch_X, batch_mask, batch_Y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, rec, F1, predictions = sess.run([self.summary_op,
                                                                 self.net.cost,
                                                                 self.net.accuracy,
                                                                 self.net.recall,
                                                                 self.net.F1_score,
                                                                 self.net.prediction],
                                                                feed_dict={self.net.x1: batch_X['left'],
                                                                           self.net.x2: batch_X['right'],
                                                                           self.net.mask1: batch_mask['left'],
                                                                           self.net.mask2: batch_mask['right'],
                                                                           self.net.y: batch_Y})
        summary_writer.add_summary(summary_str, iter)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch recall= {:.1f}%".format(iter,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            rec))

    def store_prediction(self, sess, batch_X, batch_mask, batch_Y, epoch):
        prediction = sess.run(self.net.prediction, feed_dict={self.net.x1: batch_X['left'],
                                                              self.net.x2: batch_X['right'],
                                                              self.net.mask1: batch_mask['left'],
                                                              self.net.mask2: batch_mask['right'],
                                                              self.net.y: batch_Y})
        loss, acc, rec = sess.run((self.net.cost, self.net.accuracy, self.net.recall),
                                  feed_dict={self.net.x1: batch_X['left'],
                                             self.net.x2: batch_X['right'],
                                             self.net.mask1: batch_mask['left'],
                                             self.net.mask2: batch_mask['right'],
                                             self.net.y: batch_Y})

        logging.info("Epoch {:}, Verification accuracy= {:.1f}%, recall= {:.1f}%, loss= {:.4f}").format(epoch, acc, rec,
                                                                                                        loss)
        return prediction, acc


    def stop_early(self, windows_size, val_acc ):
        """

        :param windows_size:
        :param val_acc:
        :return:
        """
        current_acc=val_acc[-1]
        stop=False

        if windows_size>=len(val_acc):
            pass
        else:
            temp_acc=val_acc[len(val_acc)-windows_size: len(val_acc)-1]
            num=sum(i>current_acc for i in temp_acc)
        if num/windows_size >=0.9:
            stop=True
        return stop







