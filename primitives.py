
import argparse
import sys
sys.path.insert(0, '../lasagne')

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
    class LSTMCell(CellLayer):
        r"""
        lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
        ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
        cell=lasagne.layers.Gate(
        W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
        outgate=lasagne.layers.Gate(),
        nonlinearity=lasagne.nonlinearities.tanh,
        hid_init=lasagne.init.Constant(0.),
        hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
        peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
        precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

        A long short-term memory (LSTM) layer.

        Includes optional "peephole connections" and a forget gate.  Based on the
        definition in [1]_, which is the current common definition.  The output is
        computed by

        .. math ::

            i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
                   + w_{ci} \odot c_{t-1} + b_i)\\
            f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
                   + w_{cf} \odot c_{t-1} + b_f)\\
            c_t &= f_t \odot c_{t - 1}
                   + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
            o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
            h_t &= o_t \odot \sigma_h(c_t)

        Parameters
        ----------
        incoming : a :class:`lasagne.layers.Layer` instance or a tuple
            The layer feeding into this layer, or the expected input shape.
        num_units : int
            Number of hidden/cell units in the layer.
        ingate : Gate
            Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
            :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
        forgetgate : Gate
            Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
            :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
        cell : Gate
            Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
            :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
        outgate : Gate
            Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
            :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
        nonlinearity : callable or None
            The nonlinearity that is applied to the output (:math:`\sigma_h`). If
            None is provided, no nonlinearity will be applied.
        cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
            Initializer for initial cell state (:math:`c_0`).
        hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
            Initializer for initial hidden state (:math:`h_0`).
        backwards : bool
            If True, process the sequence backwards and then reverse the
            output again such that the output from the layer is always
            from :math:`x_1` to :math:`x_n`.
        learn_init : bool
            If True, initial hidden values are learned.
        peepholes : bool
            If True, the LSTM uses peephole connections.
            When False, `ingate.W_cell`, `forgetgate.W_cell` and
            `outgate.W_cell` are ignored.
        gradient_steps : int
            Number of timesteps to include in the backpropagated gradient.
            If -1, backpropagate through the entire sequence.
        grad_clipping : float
            If nonzero, the gradient messages are clipped to the given value during
            the backward pass.  See [1]_ (p. 6) for further explanation.
        unroll_scan : bool
            If True the recursion is unrolled instead of using scan. For some
            graphs this gives a significant speed up but it might also consume
            more memory. When `unroll_scan` is True, backpropagation always
            includes the full sequence, so `gradient_steps` must be set to -1 and
            the input sequence length must be known at compile time (i.e., cannot
            be given as None).
        precompute_input : bool
            If True, precompute input_to_hid before iterating through
            the sequence. This can result in a speedup at the expense of
            an increase in memory usage.
        mask_input : :class:`lasagne.layers.Layer`
            Layer which allows for a sequence mask to be input, for when sequences
            are of variable length.  Default `None`, which means no mask will be
            supplied (i.e. all sequences are of the same length).
        only_return_final : bool
            If True, only return the final sequential output (e.g. for tasks where
            a single target value for the entire sequence is desired).  In this
            case, Theano makes an optimization which saves memory.

        References
        ----------
        .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
               arXiv preprint arXiv:1308.0850 (2013).
        """

        def __init__(self, incoming, num_units,
                     ingate=Gate(name='ingate'),
                     forgetgate=Gate(name='forgetgate'),
                     cell=Gate(W_cell=None, nonlinearity=tanh,
                               name='cell'),
                     outgate=Gate(name='outgate'),
                     nonlinearity=tanh,
                     cell_init=init.Constant(0.),
                     hid_init=init.Constant(0.),
                     peepholes=True,
                     grad_clipping=0,
                     **kwargs):
            super(LSTMCell, self).__init__(
                {'input': incoming},
                {'cell': cell_init, 'output': hid_init}, **kwargs)
            self.num_units = num_units
            self.peepholes = peepholes
            self.grad_clipping = grad_clipping
            if nonlinearity is None:
                self.nonlinearity = identity
            else:
                self.nonlinearity = nonlinearity

            num_inputs = np.prod(incoming.output_shape[1:])

            # Add in parameters from the supplied Gate instances
            (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
             self.nonlinearity_ingate) = ingate.add_params_to(
                self, num_inputs, num_units, step=False)

            (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
             self.nonlinearity_forgetgate) = forgetgate.add_params_to(
                self, num_inputs, num_units, step=False)

            (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
             self.nonlinearity_cell) = cell.add_params_to(
                self, num_inputs, num_units, step=False)

            (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
             self.nonlinearity_outgate) = outgate.add_params_to(
                self, num_inputs, num_units, step=False)

            # If peephole (cell to gate) connections were enabled, initialize
            # peephole connections.  These are elementwise products with the cell
            # state, so they are represented as vectors.
            if self.peepholes:
                self.W_cell_to_ingate = self.add_param(
                    ingate.W_cell, (num_units,), name='W_cell_to_ingate')

                self.W_cell_to_forgetgate = self.add_param(
                    forgetgate.W_cell, (num_units,), name='W_cell_to_forgetgate')

                self.W_cell_to_outgate = self.add_param(
                    outgate.W_cell, (num_units,), name='W_cell_to_outgate')

            # Stack input weight matrices into a (num_inputs, 4*num_units)
            # matrix, which speeds up computation
            self.W_in_stacked = self.add_param(T.concatenate(
                [self.W_in_to_ingate, self.W_in_to_forgetgate,
                 self.W_in_to_cell, self.W_in_to_outgate], axis=1),
                (num_inputs, 4 * num_units), step_only=True, precompute_input=False)

            # Same for hidden weight matrices
            self.W_hid_stacked = self.add_param(T.concatenate(
                [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
                 self.W_hid_to_cell, self.W_hid_to_outgate], axis=1),
                (num_units, 4 * num_units), step_only=True)

            # Stack biases into a (4*num_units) vector
            self.b_stacked = self.add_param(T.concatenate(
                [self.b_ingate, self.b_forgetgate,
                 self.b_cell, self.b_outgate], axis=0),
                (4 * num_units,), step_only=True, precompute_input=False)

        def get_output_shape_for(self, input_shapes):
            return {
                'cell': (input_shapes['input'][0], self.num_units),
                'output': (input_shapes['input'][0], self.num_units),
            }

        def precompute_for(self, inputs, **kwargs):
            input = inputs['input']

            # Treat all dimensions after the second as flattened feature dimensions
            if input.ndim > 3:
                input = T.flatten(input, 3)

            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, self.W_in_stacked) + self.b_stacked
            inputs['input'] = input
            return inputs

        def get_output_for(self, inputs, precompute_input=False, **kwargs):
            """
            Compute this layer's output function given a symbolic input variable

            Parameters
            ----------
            inputs : list of theano.TensorType
                `inputs[0]` should always be the symbolic input variable.  When
                this layer has a mask input (i.e. was instantiated with
                `mask_input != None`, indicating that the lengths of sequences in
                each batch vary), `inputs` should have length 2, where `inputs[1]`
                is the `mask`.  The `mask` should be supplied as a Theano variable
                denoting whether each time step in each sequence in the batch is
                part of the sequence or not.  `mask` should be a matrix of shape
                ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
                (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
                of sequence i)``. When the hidden state of this layer is to be
                pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
                should have length at least 2, and `inputs[-1]` is the hidden state
                to prefill with. When the cell state of this layer is to be
                pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
                should have length at least 2, and `inputs[-1]` is the hidden state
                to prefill with. When both the cell state and the hidden state are
                being pre-filled `inputs[-2]` is the hidden state, while
                `inputs[-1]` is the cell state.

            Returns
            -------
            layer_output : theano.TensorType
                Symbolic output variable.
            """
            input, cell_previous, hid_previous = \
                inputs['input'], inputs['cell'], inputs['output']

            # At each call to scan, input_n will be (n_time_steps, 4*num_units).
            # We define a slicing function that extract the input to each LSTM gate
            def slice_w(x, n):
                return x[:, n * self.num_units:(n + 1) * self.num_units]

            if not precompute_input:
                if input.ndim > 2:
                    input = T.flatten(input, 2)
                input = T.dot(input, self.W_in_stacked) + self.b_stacked

            # Calculate gates pre-activations and slice
            gates = input + T.dot(hid_previous, self.W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.W_cell_to_ingate
                forgetgate += cell_previous * self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)
            return {'cell': cell, 'output': hid}


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('proc_name')
    args = arg_parser.parse_args()
    globals()[args.proc_name]()

if __name__ == '__main__':
    main()
