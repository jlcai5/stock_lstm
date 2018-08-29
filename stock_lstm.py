import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import os, sys

#定义超参数 Hyperparameters


class Hp:
    batch_start = 0
    time_steps = 5
    batch_size = 20
    input_size = None
    output_size = None
    cell_size = 20
    learning_rate = 0.001
    layer_num = 3
    ckpt_dir = os.path.basename(sys.argv[0]).split(".")[0] + "/ckpt/"
    log_dir = os.path.basename(sys.argv[0]).split(".")[0] + "/logs/"
    min_cost_dir = os.path.basename(sys.argv[0]).split(".")[0] + "/"


#读取数据
def prepareData(data_file, rato=0.7):
    #定义读取excel行数
    NROWS = 1938
    df = pd.read_excel(
        data_file,
        filepath_or_buffer=data_file,
        sep=",",
        error_bad_lines=False,
        na_values="NULL",
        # usecols=[i for i in range(12)], 全都要
        nrows=NROWS,
        lineterminator="\n")
    df.dropna(inplace=True)  # 丢弃有空值的行

    df = df.sort_values("时间")  # 按照时间戳升序排序
    seq = np.array(df.values)  # 获得所有列并转为array
    seq = np.delete(seq, -1, axis=1)  # 删除时间 -1表示倒数第一列
    seq = np.delete(seq, -1, axis=1)  # 删除pred
    result = np.array(df["pred"]).reshape([-1, 1])
    xs = np.array(df["时间"]).reshape([-1, 1])
    print("seq.shape: {}".format(seq.shape))
    print("result.shape: {}".format(result.shape))
    print("xs.shape: {}".format(xs.shape))
    ## 更新Hp
    Hp.input_size = seq.shape[1]
    Hp.output_size = result.shape[1]
    # 标准化数据,零均值单位方差
    num_samples = xs.shape[0]
    standardized_seq = preprocessing.scale(seq)
    standardized_result = preprocessing.scale(result)
    split_index = int(num_samples * rato)
    x_train = standardized_seq[:split_index, :]
    y_train = standardized_result[:split_index]
    x_test = standardized_seq[split_index:, :]
    y_test = standardized_result[split_index:]
    # 标准化数据,将时间缩放到区间 [0, 1]

    # print(xs)
    min_value = np.min(xs)
    xIndex = preprocessing.maxabs_scale(xs - min_value) * num_samples
    # print(xIndex)
    return x_train, y_train, x_test, y_test, xIndex, Hp
    # seq.shape: (1000, 7)
    # result.shape: (1000, 1)
    # xs.shape: (1000, 1)


def get_batch(standardized_seq, standardized_result, standardized_xs):

    xs = standardized_xs[Hp.batch_start:Hp.batch_start +
                         Hp.time_steps * Hp.batch_size, :].reshape(
                             (Hp.batch_size, Hp.time_steps))
    #将数据转为若干最小批
    batch_seq = standardized_seq[Hp.batch_start:Hp.batch_start +
                                 Hp.time_steps * Hp.batch_size, :].reshape(
                                     [-1, Hp.time_steps, Hp.input_size])
    batch_result = standardized_result[
        Hp.batch_start:Hp.batch_start +
        Hp.time_steps * Hp.batch_size, :].reshape(
            [-1, Hp.time_steps, Hp.output_size])
    # 将时间转为若干最小批
    Hp.batch_start += Hp.time_steps
    if Hp.batch_start + Hp.time_steps * Hp.batch_size >= standardized_seq.shape[0]:
        Hp.batch_start = 0

    return batch_seq, batch_result, xs
    # batch_seq.shape: (50, 20, 7)
    # batch_result.shape: (50, 20, 1)
    # batch_xs.shape: (50, 20)


def get_test_data(x_test, y_test, Hp):

    time_step = Hp.time_steps
    size = x_test.shape[0] // time_step
    test_x, test_y = [], []
    for i in range(size - 1):
        x = x_test[i * time_step:(i + 1) * time_step, :].reshape(
            [-1, time_step, Hp.input_size])
        y = y_test[i * time_step:(i + 1) * time_step].reshape(
            [-1, time_step, Hp.output_size])
        test_x.append(x)
        test_y.append(y)
    return test_x, test_y


class LSTMRNN(object):
    def __init__(self,
                 time_steps=5,
                 input_size=None,
                 output_size=None,
                 cell_size=20,
                 batch_size=1,
                 learning_rate=0.001,
                 layer_num=2):
        self.n_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(
                tf.float32, [None, self.n_steps, self.input_size], name='xs')
            self.ys = tf.placeholder(
                tf.float32, [None, self.n_steps, self.output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.cost)

    def add_input_layer(self):

        l_in_x = tf.reshape(
            self.xs, [-1, self.input_size],
            name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([
            self.cell_size,
        ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(
            l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        #定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.cell_size, forget_bias=1.0, state_is_tuple=True)
        #添加 dropout layer, 正则化方法，可以有效防止过拟合
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell)
        #调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell] * Hp.layer_num, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            #用全零来初始化state
            self.cell_init_state = mlstm_cell.zero_state(
                self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell,
            self.l_in_y,
            initial_state=self.cell_init_state,
            time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(
            self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([
            self.output_size,
        ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            #self.pred = tf.nn.relu(tf.matmul(l_out_x, Ws_out) + bs_out)
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses')
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(
            mean=0.,
            stddev=1.,
        )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def train(x_train, y_train, index, Hp, iter_num=200):

    model = LSTMRNN(
        input_size=Hp.input_size,
        output_size=Hp.output_size,
        batch_size=Hp.batch_size)
    saver = tf.train.Saver(max_to_keep=1)
    try:
        with open(Hp.min_cost_dir + "mincost.txt", "r") as f:
            min_cost = int(f.read())
    except:
        if not os.path.exists(Hp.min_cost_dir):
            os.mkdir(Hp.min_cost_dir)
        with open(Hp.min_cost_dir + "mincost.txt", "w") as f:
            f.write("1000000")
            min_cost = 1000000
    print("min_cost:{}".format(min_cost))
    with tf.Session() as sess:
        try:
            model_file = tf.train.latest_checkpoint(Hp.ckpt_dir)
            saver.restore(sess, model_file)
            print("加载训练权重")
        except:
            print("没有训练权重")
        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(Hp.log_dir, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        plt.ion()
        plt.show()
        state = None
        for i in range(iter_num):
            seq, res, xs = get_batch(x_train, y_train, standardized_xs)
            # print(f"xs.shape{xs.shape}")
            if i == 0:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state:
                    state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [
                    model.train_op, model.cost, model.cell_final_state,
                    model.pred
                ],
                feed_dict=feed_dict)

            #plotting
            # plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :],
            #          pred.flatten()[:Hp.time_steps], 'b--')
            # plt.ylim((-2, 2))
            # plt.xlim((-10, x_train.shape[0]))
            # plt.legend(['train_res', 'train_pred'], loc='upper left')
            # plt.draw()
            # plt.title('LSTM Train')
            # plt.pause(0.01)

            if i % 200 == 0:

                print('i:{}    cost:{} '.format(i, round(cost, 4)))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
                plt.clf()  # 清屏
            if cost < min_cost:
                with open(Hp.min_cost_dir + "mincost.txt", "w") as f:
                    f.write(str(cost))
                min_cost = cost
                saver.save(sess, Hp.ckpt_dir, global_step=i + 1)


def test(x_test, y_test, Hp):

    X = tf.placeholder(tf.float32, shape=[None, Hp.time_steps, Hp.input_size])
    model_test = LSTMRNN(
        input_size=Hp.input_size, output_size=Hp.output_size, batch_size=1)

    saver = tf.train.Saver(tf.global_variables())
    seq_test, res_test = get_test_data(x_test, y_test, Hp)
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(Hp.ckpt_dir)
        print("module file : {}".format(module_file))
        saver.restore(sess, module_file)
        test_predict = []
        test_yy = []
        for step in range(len(seq_test) - 1):
            if step == 0:
                feed_dict = {
                    model_test.xs: seq_test[step],
                }
            else:
                feed_dict = {
                    model_test.xs: seq_test[step],
                    model_test.cell_init_state:
                    state  # use last state as the initial state for this run
                }
            state, prob = sess.run(
                [model_test.cell_final_state, model_test.pred],
                feed_dict=feed_dict)
            predict = prob.reshape((-1))
            print(predict)
            test_predict.extend(predict)
            test_yy.extend(res_test[step].reshape((-1)))
        test_yy = np.array(test_yy)
        test_predict = np.array(test_predict)
        acc = np.average(
            np.abs(test_predict -
                   test_yy[:len(test_predict)] / len(test_yy.tolist())))  #偏差
        print("acc : {}".format(acc))
        
        #以折线图表示结果
        plt.figure()
        plt.plot(
            list(range(len(test_predict))), test_predict.tolist(), color='b')
        plt.plot(list(range(len(test_yy))), test_yy.tolist(), color='r')
        plt.ylim((-2, 2))
        plt.legend(['test_res', 'test_pred'], loc='upper left')
        plt.title('LSTM Test')
        plt.show()


if __name__ == '__main__':
    data_file = "./btc3.xlsx"
    is_training = False
    # x_train, y_train, x_test, y_test, standardized_xs, Hp = prepareData(
    x_test, y_test, x_train, y_train, standardized_xs, Hp = prepareData(
        data_file=data_file, rato=0.3)
    if is_training:
        train(x_train, y_train, standardized_xs, Hp, iter_num=2000)
    else:
        test(x_test, y_test, Hp)
