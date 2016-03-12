
import argparse
import sys
sys.path.insert(0, '../lasagne')

import lasagne
import theano
import theano.tensor as T
from lasagne import init
from lasagne.init import *
from lasagne.layers import *
from lasagne.nonlinearities import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def seq_1_to_1():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = DenseRecurrentCell(cell_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def seq_1_to_2():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell_1 = DenseRecurrentCell(cell_inp, n_units)['output']
    cell_2 = DenseRecurrentCell(cell_inp, n_units)['output']
    cell = IdentityLayer({'output_1': cell_1, 'output_2': cell_2})
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')

    print(theano.function([l_inp.input_var], helper.get_output(l_rec))(x_in))


def seq_2_to_1():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp_1 = InputLayer((n_batch, seq_len, n_features))
    l_inp_2 = InputLayer((n_batch, seq_len, n_features))
    cell_inp_1 = InputLayer((n_batch, n_features))
    cell_inp_2 = InputLayer((n_batch, n_features))
    cell = ConcatLayer([cell_inp_1, cell_inp_2])
    cell = DenseRecurrentCell(cell, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp_1: l_inp_1, cell_inp_2: l_inp_2}, cell)

    x_in_1 = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    x_in_2 = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp_1.input_var: x_in_1, l_inp_2.input_var: x_in_2}))


def seq_0_to_1():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    hid_inp = InputLayer((n_batch, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = CustomRecurrentCell(
        None, None, DenseLayer(cell_inp, n_features), hid_init=hid_inp)['output']
    cell = DenseRecurrentCell(cell, n_units)['output']
    l_rec = RecurrentContainerLayer({}, cell, n_steps=seq_len)

    x_in = np.random.random((n_batch, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({hid_inp.input_var: x_in}))


def hid_fixed():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = DenseRecurrentCell(cell_inp, n_units, hid_init=init.Constant(0.), inits_fixed={'output'})['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def hid_learnt():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = DenseRecurrentCell(cell_inp, n_units, hid_init=init.Constant(0.))['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def hid_layer():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    hid_inp = InputLayer((n_batch, n_units))
    cell_inp = InputLayer((n_batch, n_features))
    cell = DenseRecurrentCell(cell_inp, n_units, hid_init=hid_inp)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    hid_in = np.random.random((n_batch, n_units)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in, hid_inp.input_var: hid_in}))


def vanilla_rnn():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = DenseRecurrentCell(cell_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def vanilla_lstm():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = LSTMCell(cell_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def vanilla_gru():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = GRUCell(cell_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def convolutional_rnn():
    n_batch, seq_len, n_channels, width, height = 2, 3, 4, 5, 6
    n_out_filters = 7
    filter_shape = (3, 3)

    l_inp = InputLayer((n_batch, seq_len, n_channels, width, height))
    cell_inp = InputLayer((None, n_channels, width, height))
    cell_in_to_hid = Conv2DLayer(cell_inp, n_out_filters, filter_shape, pad='same')
    cell_hid_inp = InputLayer((None, n_out_filters, width, height))
    cell_hid_to_hid = Conv2DLayer(cell_hid_inp, n_out_filters, filter_shape, pad='same')
    cell = CustomRecurrentCell(cell_inp, cell_in_to_hid, cell_hid_to_hid)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_channels, width, height)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def stack_lstm_gru():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units_1, n_units_2 = 5, 6

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = LSTMCell(cell_inp, n_units_1)['output']
    cell = GRUCell(cell, n_units_2)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec).eval({l_inp.input_var: x_in}))


def stack_lstm_gru_step_input():
    n_batch, seq_len, n_features = 2, 3, 4
    n_units_1, n_units_2 = 5, n_features

    cell_inp = InputLayer((n_batch, n_features))
    cell_hid_inp = InputLayer((n_batch, n_units_1))
    cell = LSTMCell(cell_inp, n_units_1, hid_init=cell_hid_inp)['output']
    cell = GRUCell(cell, n_units_2)['output']
    l_rec = RecurrentContainerLayer({}, cell, {cell_inp: cell}, n_steps=seq_len)

    x_in = np.random.random((n_batch, n_units_1)).astype('float32')
    print(helper.get_output(l_rec).eval({cell_hid_inp.input_var: x_in}))


def rnn_dropout_value():
    class BernoulliDropout(Layer):
        def __init__(self, incoming, n_units_, p=0.5, **kwargs):
            super().__init__(incoming, **kwargs)
            self.n_units = n_units_
            self.p = p
            self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        def get_output_for(self, input_, **kwargs):
            retain_prob = 1 - self.p
            return self._srng.binomial((input_.shape[0], self.n_units), p=retain_prob, dtype=theano.config.floatX)

    class RNNDropoutValueCell(CellLayer):
        def __init__(self, incoming, seq_incoming, n_units_, **kwargs):
            self.dropout = BernoulliDropout(seq_incoming, n_units_)
            # Passing the dropout layer to incomings directly instead of to inits will not add it to non_seqs,
            # therefore the dropout masks would change per iteration.
            super().__init__({'input': incoming}, {'output': init.Constant(0.), 'dropout': self.dropout}, **kwargs)
            self.n_units = n_units_
            n_inputs = np.prod(incoming.output_shape[1:])
            self.W_in_to_hid = self.add_param(init.Normal(0.1), (n_inputs, n_units_), name='W_in_to_hid')
            self.W_hid_to_hid = self.add_param(init.Normal(0.1), (n_units_, n_units_), name='W_hid_to_hid')

        def get_output_shape_for(self, input_shapes):
            return {'output': (input_shapes['input'][0], self.n_units)}

        def get_output_for(self, inputs, precompute_input=False, deterministic=False, **kwargs):
            input_, hid_previous, dropout_ = inputs['input'], inputs['output'], inputs['dropout']
            output = tanh(T.dot(input_, self.W_in_to_hid) + T.dot(hid_previous, self.W_hid_to_hid))
            if not deterministic:
                output = output / (1 - self.dropout.p) * dropout_
            return {'output': output}

    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = RNNDropoutValueCell(cell_inp, l_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec, deterministic=False).eval({l_inp.input_var: x_in}))


def lstm_dropout_weight():
    class LSTMDropoutWeightCell(CellLayer):
        def __init__(
            self, incoming, num_units, g_i=Gate(name='ingate'), g_f=Gate(name='forgetgate'),
            g_c=Gate(W_cell=None, nonlinearity=tanh, name='cell'), g_o=Gate(name='outgate'),
            nonlinearity=tanh, cell_init=init.Constant(0.), hid_init=init.Constant(0.), peepholes=True,
            grad_clipping=0, **kwargs
        ):
            super().__init__({
                'x': incoming, 'xi': incoming, 'xf': incoming, 'xc': incoming, 'xo': incoming,
            }, {'cell': cell_init, 'output': hid_init}, **kwargs)
            self.num_units = num_units
            self.peepholes = peepholes
            self.grad_clipping = grad_clipping
            self.nonlinearity = identity if nonlinearity is None else nonlinearity
            num_inputs = np.prod(incoming.output_shape[1:])
            self.W_xi, self.W_hi, self.b_i, self.nl_i = g_i.add_params_to(self, num_inputs, num_units)
            self.W_xf, self.W_hf, self.b_f, self.nl_f = g_f.add_params_to(self, num_inputs, num_units)
            self.W_xc, self.W_hc, self.b_c, self.nl_c = g_c.add_params_to(self, num_inputs, num_units)
            self.W_xo, self.W_ho, self.b_o, self.nl_o = g_o.add_params_to(self, num_inputs, num_units)
            if self.peepholes:
                self.W_ci = self.add_param(g_i.W_cell, (num_units,), name='W_ci')
                self.W_cf = self.add_param(g_f.W_cell, (num_units,), name='W_cf')
                self.W_co = self.add_param(g_o.W_cell, (num_units,), name='W_co')

        def get_output_shape_for(self, input_shapes):
            return {
                'cell': (input_shapes['x'][0], self.num_units),
                'output': (input_shapes['x'][0], self.num_units),
            }

        def precompute_for(self, inputs, **kwargs):
            x = inputs['x']
            if x.ndim > 3:
                x = T.flatten(x, 3)
            return {
                'x': x,
                'xi': T.dot(x, self.W_xi) + self.b_i,
                'xf': T.dot(x, self.W_xf) + self.b_f,
                'xc': T.dot(x, self.W_xc) + self.b_c,
                'xo': T.dot(x, self.W_xo) + self.b_o,
            }

        def get_output_for(self, inputs, precompute_input=False, **kwargs):
            if not precompute_input:
                raise NotImplementedError
            c_tm1, h_tm1 = inputs['cell'], inputs['output']
            i_t = inputs['xi'] + T.dot(h_tm1, self.W_hi)
            f_t = inputs['xf'] + T.dot(h_tm1, self.W_hf)
            c_t = inputs['xc'] + T.dot(h_tm1, self.W_hc)
            o_t = inputs['xo'] + T.dot(h_tm1, self.W_ho)
            if self.grad_clipping:
                i_t = theano.gradient.grad_clip(i_t, -self.grad_clipping, self.grad_clipping)
                f_t = theano.gradient.grad_clip(f_t, -self.grad_clipping, self.grad_clipping)
                c_t = theano.gradient.grad_clip(c_t, -self.grad_clipping, self.grad_clipping)
                o_t = theano.gradient.grad_clip(o_t, -self.grad_clipping, self.grad_clipping)
            if self.peepholes:
                i_t += c_tm1 * self.W_ci
                f_t += c_tm1 * self.W_cf
            i_t = self.nl_i(i_t)
            f_t = self.nl_f(f_t)
            c_t = self.nl_c(c_t)
            c_t = f_t * c_tm1 + i_t * c_t
            if self.peepholes:
                o_t += c_t * self.W_co
            o_t = self.nl_o(o_t)
            h_t = o_t * self.nonlinearity(c_t)
            return {'cell': c_t, 'output': h_t}

    lasagne.random.get_rng().seed(1234)

    n_batch, seq_len, n_features = 2, 3, 4
    n_units = 5

    l_inp = InputLayer((n_batch, seq_len, n_features))
    cell_inp = InputLayer((n_batch, n_features))
    cell = LSTMDropoutWeightCell(cell_inp, n_units)['output']
    l_rec = RecurrentContainerLayer({cell_inp: l_inp}, cell)

    x_in = np.random.random((n_batch, seq_len, n_features)).astype('float32')
    print(helper.get_output(l_rec, deterministic=False).eval({l_inp.input_var: x_in}))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('proc_name')
    args = arg_parser.parse_args()
    globals()[args.proc_name]()

if __name__ == '__main__':
    main()
