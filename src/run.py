import tensorflow as tf
import os
import src.preprocessing as precessing
import src.siamese_RNN as siamese_RNN
import src.train as train

baseDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
train_file=baseDir+'/data/raw_data/atec_nlp_sim_train_processed.csv'
userdict=baseDir+'/data/extra_dict/dict.txt'
pre_w2vmodel=baseDir+'/data/extra_dict/Word60.model'
output_path='/data/model/'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'the batch_size of the training procedure')
flags.DEFINE_float('lr', 0.0001, 'the learning rate')
flags.DEFINE_float('decay_rate', 0.95, 'the learning rate decay')
flags.DEFINE_integer('emdedding_dim', 60, 'embedding dim')
flags.DEFINE_integer('hidden_neural_size', 50, 'LSTM hidden neural size')
flags.DEFINE_integer('max_len', 73, 'max_len of training sentence')
flags.DEFINE_integer('valid_num', 100, 'epoch num of validation')
flags.DEFINE_float('frac', 0.9, 'fraction of training data')
flags.DEFINE_integer('num_epoch', 360, 'num epoch')
flags.DEFINE_integer('max_decay_epoch', 100, 'num epoch')
flags.DEFINE_string('baseDir', baseDir, 'root directory of project')
flags.DEFINE_string('train_file', train_file, 'file name for training')
flags.DEFINE_string('userdict', userdict, 'extra dict from user')
flags.DEFINE_string('w2v', pre_w2vmodel, 'word2vec model')
flags.DEFINE_string('cell', 'LSTM', 'RNN cell')
flags.DEFINE_string('output_path', output_path, 'path to save model file')
flags.DEFINE_boolean('restore', False, 'False: retrain the model, True: restore the previous trained model')




def main():
    # processing the data
    data=precessing.load_data(FLAGS.train_file)
    data=precessing.cut_word(data, userdict)
    pre_embedding_index, vector_size=precessing.get_embedding_index(FLAGS.w2v)
    if vector_size is not None:
        embedding_dim=vector_size
    else:
        embedding_dim=FLAGS.emdedding_dim
    vocabulary=precessing.My_Vocabulary(data, pre_embedding_index, embedding_dim, FLAGS.max_len)
    embedding_matrix=vocabulary.get_embedding_matrix()
    X, mask, Y=vocabulary.padding_and_masking()

    # define a siamese RNN
    net=siamese_RNN.Siamese_RNN(embedding_matrix, FLAGS.hidden_neural_size,  cell=FLAGS.cell, max_length=FLAGS.max_len )

    trainer=train.Trainner(net, FLAGS.batch_size)

    trainer.train(output_path, X, mask, Y, "", frac=FLAGS.frac, epochs=FLAGS.num_epoch, restore=FLAGS.restore)


if __name__ == "__main__":
    tf.app.run()









