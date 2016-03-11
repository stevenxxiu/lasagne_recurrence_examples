
import argparse
import sys
sys.path.insert(0, '../lasagne')

import numpy as np
import theano
from lasagne import init
from lasagne.layers import (
    CellLayer, ConcatLayer, CustomRecurrentCell, CustomRecurrentLayer, DenseLayer, DenseRecurrentCell, Gate, GRUCell,
    GRULayer, IdentityLayer, InputLayer, Layer, LSTMCell, LSTMLayer, RecurrentContainerLayer, RecurrentLayer,
    ReshapeLayer, helper
)


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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('proc_name')
    args = arg_parser.parse_args()
    globals()[args.proc_name]()

if __name__ == '__main__':
    main()
