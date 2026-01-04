import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attention import TransformerBlock


def obs_to_tensor(obs, device):
    # box_num = obs.box_num[0]
    box_num = obs.box_num.item()
    # ems_num = obs.ems_num[0]
    ems_num = obs.ems_num.item()
    # state_num = obs.state_num[0]
    state_num = obs.state_num.item()
    # container_width = int(obs.container_width[0])
    # container_length = int(obs.container_length[0])
    container_width = int(obs.container_width.item())
    container_length = int(obs.container_length.item())

    # corner_num = obs.corner_num[0]
    corner_num = obs.corner_num.item()

    box_state_num = box_num * state_num

    # batch_size = len(obs.box_num)
    batch_size = 1

    if len(obs.box_states.shape) == 3:
        # box_num = box_num[0]
        # ems_num = ems_num[0]
        # state_num = state_num[0]
        # box_state_num = box_state_num[0]
        box_num = box_num
        ems_num = ems_num
        state_num = state_num
        box_state_num = box_state_num

    box_states = obs.box_states.reshape(batch_size, box_state_num, 3)
    
    pre_box = obs.pre_box.reshape(batch_size, 3)
    heightmap = obs.heightmap.reshape(batch_size, 2, container_width, container_length)

    valid_mask = obs.valid_mask.reshape(batch_size, -1)
    access_mask = obs.access_mask.reshape(batch_size, -1)
    
    ems = obs.ems.reshape(batch_size, ems_num, -1)
    ems_mask = obs.ems_mask.reshape(batch_size, ems_num)
    
    ems_size_mask = obs.ems_size_mask.reshape(batch_size, ems_num, box_state_num)
    ems_to_box_mask = obs.ems_to_box_mask.reshape(batch_size, corner_num, ems_num, box_state_num)

    # if len(obs.precedence[0]) > 1:
    if len(obs.precedence) > 1:
        # NOTE prec
        prec_dim = 2
        prec_states = obs.precedence.reshape(batch_size, box_state_num, box_num, prec_dim)
        prec_states = torch.tensor(prec_states).float().to(device)
    else:
        prec_states = None
    
    pre_box = torch.tensor(pre_box).float().to(device)
    heightmap = torch.tensor(heightmap).float().to(device)
        
    box_states = torch.tensor(box_states).to(device)
    valid_mask = torch.tensor(valid_mask).to(device)
    access_mask = torch.tensor(access_mask).to(device)
    ems = torch.tensor(ems).to(device)
    ems_mask = torch.tensor(ems_mask).to(device)
    ems_size_mask = torch.tensor(ems_size_mask).to(device)
    ems_to_box_mask = torch.tensor(ems_to_box_mask).to(device)
    
    ems_mask = ems_mask.unsqueeze(1).unsqueeze(2)

    return ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, pre_box, heightmap


class RnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RnnEncoder, self).__init__()

        self.gru = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        self.drop_hh = nn.Dropout(p=dropout)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

    def forward(self, data):
        # batch_size x data_num x dim_num
        # output  batch_size x (hidden_size) x dim_num

        # encoder_input  batch_size x data_num x dim_num
        batch_size = data.shape[0]
        data_num = data.shape[1]
        dim_num = data.shape[2]

        output = torch.zeros( batch_size, self.hidden_size, dim_num ).to(self.device)
        
        for dim_index in range(dim_num):            
            # dim_input  batch_size x data_num x input_size(1)
            dim_input = data[:,:,dim_index:dim_index+1]
            last_hh = None
            rnn_out, last_hh = self.gru(dim_input, last_hh)
            
            if self.num_layers == 1:
                # If > 1 layer dropout is already applied
                last_hh = self.drop_hh(last_hh)
            output[:,:,dim_index] = last_hh
            
        # output  batch_size x hidden_size x dim_num
        return output
        


class PrecedenceModule(nn.Module):
    def __init__(self, prec_dim, embed_size, prec_type, device, heads=4, dropout=0 ):
        super(PrecedenceModule, self).__init__()
        self.prec_type = prec_type
        self.prec_dim = prec_dim
        self.device = device

        if prec_type == 'attn':
            # self.position_embedding = nn.Embedding(1000, embed_size)
            self.prec_embedding = nn.Linear(prec_dim, embed_size)
            self.attn = TransformerBlock(embed_size, heads, dropout, 4)
        elif prec_type == 'cnn':
            self.prec_embedding = nn.Conv1d(prec_dim, embed_size, kernel_size=1)
        elif prec_type == 'rnn':
            self.prec_embedding = RnnEncoder(1, embed_size, 1, 0.1, device)

    def forward(self, precedence, top_mask):
        '''
            prec_states: [batch x (rot * axis * box_num)  x box_num x 2]
            top_mask: [batch x box_num]  a valid mask, if box *already* packed, no compute in atten
            
            ret: [batch x (rot * axis * box_num) x embed_size]
        '''
        batch_size, state_num, box_num, _ = precedence.shape
        mask = top_mask.unsqueeze(1).unsqueeze(2)

        if self.prec_type == 'attn':
            prec_vecs = self.prec_embedding(precedence)
            
            box_idx = [ i % box_num for i in range(state_num) ]
            state_idx = [ i for i in range(state_num) ]

            bs_mask = mask.expand([batch_size, state_num, 1, box_num]).reshape(batch_size * state_num, 1, 1, box_num)
            values = prec_vecs.view(batch_size * state_num, box_num, -1)
            query = prec_vecs[:, state_idx, box_idx].view(batch_size * state_num, 1, -1)

            ret = self.attn(values, values, query, bs_mask)
            ret = ret.view(batch_size, state_num, -1)
            
            # NOTE old way to compute >> 

            # ret = []
            # for i in range(state_num):
            #     prec_vec = prec_vecs[:,i]
            #     # positions = torch.arange(0, box_num).expand(batch_size, box_num).to(self.device)
            #     # emb_pos = self.position_embedding(positions)
            #     # prec_vec = prec_vec + emb_pos
            #     prec_i = i % box_num
            #     prec_vec = self.attn(prec_vec, prec_vec, prec_vec[:, prec_i:prec_i+1], mask )
            #     ret.append(prec_vec)
            # ret = torch.cat(ret, dim=1)

        elif self.prec_type == 'cnn' or self.prec_type == 'rnn':
            prec_vecs = precedence.reshape(batch_size, state_num, -1)
            prec_vecs = prec_vecs.transpose(2,1)
            ret = self.prec_embedding( prec_vecs ).transpose(2,1)

        return ret


class HeightmapEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, map_size):
        super(HeightmapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, int(hidden_size/4), stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(int(hidden_size/4), int(hidden_size/2), stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(int(hidden_size/2), int(hidden_size), kernel_size=( math.ceil(map_size[0]/4), math.ceil(map_size[1]/4) ) )

    def forward(self, input):
        output = F.leaky_relu(self.conv1(input))
        output = F.leaky_relu(self.conv2(output))
        output = self.conv3(output).squeeze(-1)
        return output  # (batch, hidden_size, seq_len)

class SpaceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, corner_num=1) -> None:
        super(SpaceEncoder, self).__init__()

        self.corner_num = corner_num

        if input_dim == 100:
            input_dim -= 1

            type_dim = int(hidden_dim / 4)
            self.ems_type_embedding = nn.Linear(1, type_dim)
            self.ems_merge = nn.Linear(hidden_dim + type_dim, hidden_dim)

            if corner_num > 1:
                input_dim += 1
            
            self.ems_embedding = nn.Sequential(
                # nn.Linear(input_dim, input_dim-1),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.ems_type_embedding = None

            if corner_num > 1:
                input_dim += 1
            
            self.ems_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def embed(self, ems_data):
        if self.ems_type_embedding is None:
            ret = self.ems_embedding(ems_data)
        else:
            type_vec = self.ems_type_embedding(ems_data[:, :, 6:7])
            ems_vec = self.ems_embedding(ems_data[:, :, :6])
            ret = self.ems_merge( torch.cat([ems_vec, type_vec], dim=-1) )
        return ret

    def forward(self, ems):
        ems_vecs = []
        for ems_i in range(self.corner_num):
            # pos | size
            ems_in = ems.clone()
            # ems_size = ems_in[:, :, 3:6]
            
            
            if self.corner_num == 1:
                ems_vec = self.embed(ems_in)
            else:
                ems_ids = torch.zeros_like(ems[:, :, :1]) + ems_i
                # if ems_i == 1:
                #     ems_in[:,:,0] += ems_size[:,:,0]
                # elif ems_i == 2:
                #     ems_in[:,:,1] += ems_size[:,:,1]
                # elif ems_i == 3:
                #     ems_in[:,:,0] += ems_size[:,:,0]
                #     ems_in[:,:,1] += ems_size[:,:,1]
                # elif ems_i == 2:
                #     ems_in[:,:,2] += (ems_size[:,:,2] - 1)
                # elif ems_i == 3:
                #     ems_in[:,:,0] += (ems_size[:,:,0] - 1)
                #     ems_in[:,:,2] += (ems_size[:,:,2] - 1)
                ems_vec = self.embed(torch.cat([ems_in, ems_ids], dim=-1))
            ems_vecs.append(ems_vec)

        return ems_vecs

class ObjectEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, prec_dim=2, prec_type='none', \
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(ObjectEncoder, self).__init__()

        self.prec_type = prec_type
        self.device = device

        if self.prec_type == 'none':
            self.box_embedding = nn.Linear(input_dim, hidden_dim)
            self.prec_embeding = None
        else:
            half_dim = int(hidden_dim/2)
            self.box_embedding = nn.Linear(input_dim, half_dim)
            self.prec_embeding = PrecedenceModule(prec_dim, half_dim, prec_type, device=device)
            # elif self.prec_type == 'rnn':
            #     self.prec_embeding = RnnEncoder(1, half_size, 1, 0.1)
            # elif self.prec_type == 'cnn':
            #     self.prec_embeding = Encoder(box_num * 3, half_size)

    def forward(self, box_states, prec_states=None, valid_mask=None):
        '''
        box_states:  [batch x (rot * axis * box_num)  x 3]
        prec_states: [batch x (rot * axis * box_num)  x box_num x 2]
        valid_mask:  [batch x (rot * axis * box_num)]
        '''

        if self.prec_embeding is None:
            box_vecs = self.box_embedding(box_states)
        else:
            # precedences = precedences.clone()
            batch_size, state_num, box_num, _ = prec_states.shape
            if valid_mask is None:
                prec_mask = torch.ones(batch_size, box_num).to(self.device)
            else:
                # NOTE prec_mask, the object which are not in remove_list, we only need the move prec, for attn
                prec_mask = valid_mask.reshape(batch_size, 2, -1, box_num )[:,0, 0]

            box_vec = self.box_embedding( box_states )
            prec_vec = self.prec_embeding( prec_states, prec_mask )
            # prec_vec = self.prec_embeding( prec_states ).transpose(1,2) # cnn, rnn
            box_vecs = torch.cat([box_vec, prec_vec], dim=2)

        return box_vecs


