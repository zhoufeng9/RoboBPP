import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossTransformer, CrossLayer
from .encoder import obs_to_tensor, ObjectEncoder, SpaceEncoder

class StrategyAttention(nn.Module):
    def __init__(self, encoder_type, box_dim, ems_dim, hidden_dim, prec_dim, corner_num, stable_predict=False, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(StrategyAttention, self).__init__()
        self.encoder_type = encoder_type
        self.box_dim = box_dim
        self.device = device

        self.corner_num = corner_num

        # prec_dim = 2
        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, prec_dim, encoder_type, device)
        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim, corner_num)

        self.transformer = CrossTransformer(hidden_dim, mask_predict=stable_predict, device=device, corner_num=corner_num)

    def forward(self, box, precedences, ems, ems_mask, ems_to_box_mask, box_valid_mask=None):
        
        box_vecs = self.obj_encoder(box, precedences, box_valid_mask)
        ems_vecs = self.space_encoder(ems)
        attn_vecs = self.transformer(ems_vecs, box_vecs, ems_mask, box_valid_mask, ems_to_box_mask)

        return attn_vecs

class Net(nn.Module):

    def __init__(self, box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict=False,
                corner_num = 1,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Net, self).__init__()

        self.corner_num = corner_num

        self.strategy = StrategyAttention( encoder_type, box_dim, ems_dim, hidden_dim, prec_dim, corner_num, stable_predict, device )

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


        self.device = device
        self.box_dim = box_dim
        
        self.encoder_type = encoder_type

    def forward(self, obs, state=None, info={} ):
        
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        attn_vecs = self.strategy(box_states, prec_states, ems, ems_mask, ems_to_box_mask, valid_mask)

        # mask the attnetion score
        if attn_vecs.shape[1] == ems_num:
            # attn_score = attn_vecs + precedence_mask.unsqueeze(1).log()
            attn_score = attn_vecs + (valid_mask * access_mask).unsqueeze(1).float().log()
        else:
            attn_vecs = attn_vecs.reshape(batch_size, -1, ems_num, box_state_num)
            attn_score = attn_vecs.clone()
            for i in range(attn_vecs.shape[1]):
                # attn_score[:,i,:,:] = attn_vecs[:,i,:,:] + precedence_mask.unsqueeze(1).log()
                attn_score[:,i,:,:] = attn_vecs[:,i,:,:] + (valid_mask * access_mask).unsqueeze(1).float().log()

        attn_score = attn_score.reshape(batch_size, -1)
        probs = F.softmax(attn_score, dim=1)
        
        # if self.training == False:
        #     prob_max = probs.max(dim=1, keepdim=True)[0]
        #     probs[ probs != prob_max ] = 0
        #     probs[ probs == prob_max ] = 1
        #     probs /= probs.sum(dim=1, keepdim=True)
        
        return probs, state

class Critic(nn.Module):

    def __init__(self, box_dim, ems_dim, box_num, ems_num, hidden_dim, prec_dim=2, heads = 4, output_dim=1,
                 prec_type = 'none',
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Critic, self).__init__()
        
        self.box_num = box_num
        self.ems_num = ems_num

        # prec_dim = 2
        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, prec_dim, prec_type, device=device)
        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim)

        # self.box_attn = AttnModule(hidden_dim, heads, 0, device, 50)
        # self.ems_attn = AttnModule(hidden_dim, heads, 0, device, 50)
        
        self.box_combine_mlp = nn.Sequential(
            nn.Linear(box_num, box_num),
            nn.ReLU(),
            nn.Linear(box_num, 1)
        )
        
        self.ems_combine_mlp = nn.Sequential(
            nn.Linear(ems_num, ems_num),
            nn.ReLU(),
            nn.Linear(ems_num, 1)
        )

        # self.output_attn = SelfAttention(hidden_dim, heads)
        self.output_layer = nn.Sequential(
            # nn.Linear(hidden_dim*2, 1),
            # nn.ReLU()
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.device = device

    def forward(self, obs, act=None, info={} ):
        
        # encoder_input, decoder_input = observation
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)
        
        # TODO what

        valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        # ems_mask = ems_mask.unsqueeze(1).unsqueeze(2)

        box_vecs = self.obj_encoder(box_states, prec_states, valid_mask)
        ems_vecs = self.space_encoder(ems)[0]

        # box_vecs = self.box_attn( box_vecs, valid_mask, pos_embed=False)
        # ems_vecs = self.ems_attn( ems_vecs, ems_mask, pos_embed=False)

        box_feats = box_vecs.masked_fill(valid_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))
        ems_feats = ems_vecs.masked_fill(ems_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))

        box_vec = self.box_combine_mlp(box_feats.transpose(2,1)).squeeze(2)
        ems_vec = self.ems_combine_mlp(ems_feats.transpose(2,1)).squeeze(2)
        
        output_value = self.output_layer( torch.cat([box_vec, ems_vec], dim=-1) )
        # output_value = F.relu(output_value)
        return output_value

class Actor(nn.Module):
    def __init__(self, box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device) -> None:
        super(Actor, self).__init__()
        self.actor = Net(box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device)
    
    def forward(self, obs, state=None, info={} ):
        probs, state = self.actor(obs, state=None, info={})
        return probs, state