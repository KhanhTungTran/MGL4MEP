import torch
import torch.nn as nn
import torch.nn.functional as F


class MGL4MEP_S(nn.Module):
    def __init__(self, args):
        super(MGL4MEP_S, self).__init__()
        self.hidden_size = 130
        self.num_layers = args.lstm_num_layers
        self.lstm = nn.LSTM(input_size=args.s_input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.single_mode = args.single_mode
        if args.single_mode:
            self.fc = nn.Linear(self.hidden_size, args.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_size, args.output_dim*args.horizon)
        self.horizon = args.horizon
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        s_and_r_inputs = inputs['s_and_r']
        # print('INPUT: ', s_and_r_inputs.shape)
        if s_and_r_inputs.shape[-1] != 1:
            s_inputs = s_and_r_inputs[..., :-1]
        else:
            s_inputs = s_and_r_inputs
        # print('INPUT AFTER: ', s_inputs.shape)
        n_samples = s_and_r_inputs.shape[0]
        h_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_inputs.device)
        c_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_inputs.device)

        h1, (h1_T, c1_T) = self.lstm(s_inputs, (h_t, c_t))
        s_output = h1[:, -1, :].squeeze()
        if len(s_output.shape) == 1:
            s_output = s_output.unsqueeze(dim=0)
        final_classifer_input = s_output
        h2 = self.fc(final_classifer_input)
        if self.act:
            output = self.act(h2)
        else:
            output = h2
        if not self.single_mode:
            output = output.reshape(output.shape[0], self.horizon, 2)
        return output


class MGL4MEP_SR(nn.Module):
    def __init__(self, args):
        super(MGL4MEP_SR, self).__init__()
        self.hidden_size = 130
        self.num_layers = args.lstm_num_layers
        self.lstm = nn.LSTM(input_size=args.sr_input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.single_mode = args.single_mode
        self.output_dim = args.output_dim
        if args.single_mode:
            self.fc = nn.Linear(self.hidden_size, self.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_dim*args.horizon)
        self.horizon = args.horizon
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        s_and_r_inputs = inputs['s_and_r']
        n_samples = s_and_r_inputs.shape[0]
        h_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_and_r_inputs.device)
        c_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_and_r_inputs.device)

        h1, (h1_T, c1_T) = self.lstm(s_and_r_inputs, (h_t, c_t))
        s_and_r_output = h1[:, -1, :].squeeze()
        if len(s_and_r_output.shape) == 1:
            s_and_r_output = s_and_r_output.unsqueeze(dim=0)
        final_classifer_input = s_and_r_output
        h2 = self.fc(final_classifer_input)
        output = self.act(h2)
        if not self.single_mode:
            output = output.reshape(output.shape[0], self.horizon, self.output_dim)
        return output


class MGL4MEP_SRE(nn.Module):
    def __init__(self, args):
        super(MGL4MEP_SRE, self).__init__()
        self.hidden_size = 130
        self.num_layers = args.lstm_num_layers
        self.lstm = nn.LSTM(input_size=args.sr_input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.single_mode = args.single_mode
        if args.single_mode:
            self.fc = nn.Linear(self.hidden_size+args.rnn_units, args.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_size+args.rnn_units, args.output_dim*args.horizon)
        self.horizon = args.horizon
        self.act = nn.Sigmoid()
        self.graph_encoder = AGCRN(args)

    def forward(self, inputs):
        graph_inputs = inputs['entity']
        if 'entity_mask' in inputs.keys():
            graph_adj_mask = inputs['entity_mask']
            graph_output = self.graph_encoder(graph_inputs, graph_adj_mask)
        else:
            graph_output = self.graph_encoder(graph_inputs)
        graph_output = torch.max(graph_output, dim=2)[0] # TAKING MAX HERE
        graph_output = torch.reshape(graph_output, (graph_output.shape[0], -1))

        s_and_r_inputs = inputs['s_and_r']
        n_samples = s_and_r_inputs.shape[0]
        h_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_and_r_inputs.device)
        c_t = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32).to(s_and_r_inputs.device)

        h1, (h1_T, c1_T) = self.lstm(s_and_r_inputs, (h_t, c_t))
        s_and_r_output = h1[:, -1, :].squeeze()
        if len(s_and_r_output.shape) == 1:
            s_and_r_output = s_and_r_output.unsqueeze(dim=0)

        final_classifer_input = torch.cat((s_and_r_output, graph_output), dim=-1)
        h2 = self.fc(final_classifer_input)
        output = self.act(h2)
        if not self.single_mode:
            output = output.reshape(output.shape[0], self.horizon, 2)
        return output


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings, adj_mask):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N], adj_mask shaped [B, N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        supports = supports.unsqueeze(0).repeat(adj_mask.shape[0], 1, 1)
        supports = supports * adj_mask
        ident_mat = torch.eye(node_num).unsqueeze(0).repeat(adj_mask.shape[0], 1, 1).to(supports.device)
        support_set = [ident_mat, supports]
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("kbnm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, adj_mask):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        #adj_mask: B, num_nodes, num_nodes
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, adj_mask))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, adj_mask))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, adj_mask):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        #shape of node_embeddings: (N, D)
        #shape of adj_mask: (B, T, N, N)

        # print("SHAPE OF INPUT: ", x.shape)
        # print("NUMBER OF NODE: ", self.node_num)
        # print("INPUT DIMENSION: ", self.input_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, adj_mask[:, t, :, :])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source, adj_mat=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)
        if adj_mat is None:
            adj_mat = torch.ones((source.shape[0], source.shape[1], source.shape[2], source.shape[2]))
        adj_mat = adj_mat.to(source.device)
        output, _ = self.encoder(source, init_state, self.node_embeddings, adj_mat)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        '''
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        '''
        return output
