"""A pure python LSTM implementation that supports double differentiation"""
import torch as th
from torch import nn


class MyLSTMCell(nn.LSTMCell):

    def forward(self, x, hx=None):
        self.check_forward_input(x)
        if hx is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(x, hx[0], '[0]')
        self.check_forward_hidden(x, hx[1], '[1]')
        # state
        h, c = hx
        # Get gates
        gates = th.einsum("bd,hd->bh", [h, self.weight_hh]) + \
            th.einsum("bd,hd->bh", [x, self.weight_ih])
        if self.bias_hh is not None:
            gates += (self.bias_ih + self.bias_hh).unsqueeze(0)
        # Now get each gate
        dh = self.hidden_size
        gate_i = th.sigmoid(gates[:, :dh])
        gate_f = th.sigmoid(gates[:, dh:(2*dh)])
        gate_g = th.tanh(gates[:, (2*dh):(3*dh)])
        gate_o = th.sigmoid(gates[:, (3 * dh):])

        new_c = gate_f * c + gate_g * gate_i
        new_h = gate_o * th.tanh(new_c)

        return new_h, new_c


class MyLSTM(nn.Module):

    def __init__(
        self,
        num_layers=1,
        input_size=2,
        hidden_size=64,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        dims = [input_size] + [hidden_size] * num_layers
        self.fwd_cells = nn.ModuleList([
            MyLSTMCell(input_size=di, hidden_size=do)
            for di, do in zip(dims[:-1], dims[1:])
        ])
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.bwd_cells = nn.ModuleList([
                MyLSTMCell(input_size=di, hidden_size=do)
                for di, do in zip(dims[:-1], dims[1:])
            ])

        self.n_directions = 2 if bidirectional else 1

    def transduce(self, inputs, h0, c0, lengths=None, forward=True):
        if not forward and not self.bidirectional:
            raise ValueError("Uhh...")
        # Sentence lengths
        min_length = inputs.size(0)
        if lengths is not None:
            if not isinstance(lengths, th.LongTensor):
                lengths = th.LongTensor(lengths).to(inputs.device)
            min_length = th.min(lengths).item()
        # Dimensions
        L, bsz, d = inputs.size()
        h, c = h0, c0
        hs, cs = [], []
        for i in range(L):
            hs_i, cs_i = [], []
            # Reset the state
            if lengths is not None and not forward and i <= L - min_length:
                # mask is 1 if i<=L-length (we need to rest the state)
                mask = lengths.le(L - i).view(1, bsz, 1).to(h.device)
                h = th.where(mask, h0, h)
                c = th.where(mask, c0, c)
            for layer in range(self.num_layers):
                # Select this layer's state
                hl = h[layer]
                cl = c[layer]
                # Input
                if layer == 0:
                    x = inputs[i] if forward else inputs[L - i - 1]
                else:
                    x = hs_i[-1]
                # Choose LSTM cell
                if forward:
                    cell = self.fwd_cells[layer]
                else:
                    cell = self.bwd_cells[layer]
                # Reset states
                hl, cl = cell(x, hx=(hl, cl))
                hs_i.append(hl)
                cs_i.append(cl)
            h, c = th.stack(hs_i, dim=0), th.stack(cs_i, dim=0)
            hs.append(h)
            cs.append(c)
        if not forward:
            hs = hs[::-1]
        all_out = th.stack(hs)[:, -1, :, :]
        return all_out

    def forward(self, inputs, h0, lengths=None):
        # Handle packed sequence
        was_packed = False
        if isinstance(inputs, nn.utils.rnn.PackedSequence):
            inputs, lengths = nn.utils.rnn.pad_packed_sequence(inputs)
            was_packed = True
        # Dimensions
        L, bsz, d = inputs.size()
        # Initial state
        tot_layers = self.num_layers * self.n_directions
        if h0 is None:
            h0 = inputs.new_zeros((tot_layers, bsz, self.hidden_size))
            c0 = inputs.new_zeros((tot_layers, bsz, self.hidden_size))
        else:
            h0, c0 = h0
            if h0.size(1) != bsz:
                h0 = h0.repeat(1, bsz, 1)
            if c0.size(1) != bsz:
                c0 = c0.repeat(1, bsz, 1)
        # Forward transduction
        out = self.transduce(
            inputs,
            h0=h0[:self.num_layers],
            c0=c0[:self.num_layers],
            lengths=lengths,
            forward=True
        )
        # Backward transduction
        if self.bidirectional:
            out_bwd = self.transduce(
                inputs,
                h0=h0[self.num_layers:],
                c0=c0[self.num_layers:],
                lengths=lengths,
                forward=False,
            )
            out = th.cat([out, out_bwd], dim=-1)
        # Handle packed sequence
        if was_packed:
            out = nn.utils.rnn.pack_padded_sequence(
                out, lengths, enforce_sorted=False)
        return out, None

    def to_cudnn_bilstm(self):
        cudnn_bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        for layer in range(self.num_layers):
            # Fwd ih
            cudnn_W_ih = getattr(cudnn_bilstm, f"weight_ih_l{layer}")
            cudnn_W_ih.data = self.fwd_cells[layer].weight_ih.data.clone()
            cudnn_b_ih = getattr(cudnn_bilstm, f"bias_ih_l{layer}")
            cudnn_b_ih.data = self.fwd_cells[layer].bias_ih.data.clone()
            # Fwd hh
            cudnn_W_hh = getattr(cudnn_bilstm, f"weight_hh_l{layer}")
            cudnn_W_hh.data = self.fwd_cells[layer].weight_hh.data.clone()
            cudnn_b_hh = getattr(cudnn_bilstm, f"bias_hh_l{layer}")
            cudnn_b_hh.data = self.fwd_cells[layer].bias_hh.data.clone()
            # Bwd ih
            cudnn_W_ih = getattr(cudnn_bilstm, f"weight_ih_l{layer}_reverse")
            cudnn_W_ih.data = self.bwd_cells[layer].weight_ih.data.clone()
            cudnn_b_ih = getattr(cudnn_bilstm, f"bias_ih_l{layer}_reverse")
            cudnn_b_ih.data = self.bwd_cells[layer].bias_ih.data.clone()
            # Bwd hh
            cudnn_W_hh = getattr(cudnn_bilstm, f"weight_hh_l{layer}_reverse")
            cudnn_W_hh.data = self.bwd_cells[layer].weight_hh.data.clone()
            cudnn_b_hh = getattr(cudnn_bilstm, f"bias_hh_l{layer}_reverse")
            cudnn_b_hh.data = self.bwd_cells[layer].bias_hh.data.clone()
        cudnn_bilstm.flatten_parameters()
        return cudnn_bilstm

    @staticmethod
    def from_cudnn_bilstm(cudnn_bilstm):
        bilstm = MyLSTM(
            input_size=cudnn_bilstm.input_size,
            hidden_size=cudnn_bilstm.hidden_size,
            num_layers=cudnn_bilstm.num_layers,
            bidirectional=True,
        )
        for layer in range(bilstm.num_layers):
            # Fwd ih
            cudnn_W_ih = getattr(cudnn_bilstm, f"weight_ih_l{layer}")
            bilstm.fwd_cells[layer].weight_ih.data = cudnn_W_ih.data.clone()
            cudnn_b_ih = getattr(cudnn_bilstm, f"bias_ih_l{layer}")
            bilstm.fwd_cells[layer].bias_ih.data = cudnn_b_ih.data.clone()
            # Fwd hh
            cudnn_W_hh = getattr(cudnn_bilstm, f"weight_hh_l{layer}")
            bilstm.fwd_cells[layer].weight_hh.data = cudnn_W_hh.data.clone()
            cudnn_b_hh = getattr(cudnn_bilstm, f"bias_hh_l{layer}")
            bilstm.fwd_cells[layer].bias_hh.data = cudnn_b_hh.data.clone()
            # Bwd ih
            cudnn_W_ih = getattr(cudnn_bilstm, f"weight_ih_l{layer}_reverse")
            bilstm.bwd_cells[layer].weight_ih.data = cudnn_W_ih.data.clone()
            cudnn_b_ih = getattr(cudnn_bilstm, f"bias_ih_l{layer}_reverse")
            bilstm.bwd_cells[layer].bias_ih.data = cudnn_b_ih.data.clone()
            # Bwd hh
            cudnn_W_hh = getattr(cudnn_bilstm, f"weight_hh_l{layer}_reverse")
            bilstm.bwd_cells[layer].weight_hh.data = cudnn_W_hh.data.clone()
            cudnn_b_hh = getattr(cudnn_bilstm, f"bias_hh_l{layer}_reverse")
            bilstm.bwd_cells[layer].bias_hh.data = cudnn_b_hh.data.clone()
        return bilstm
