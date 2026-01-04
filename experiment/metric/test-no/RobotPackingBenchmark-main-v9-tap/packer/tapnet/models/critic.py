from tapnet.models.network import Actor
from tapnet.models.encoder import obs_to_tensor, SpaceEncoder
from tapnet.models.attention import AttnModule, SelfAttention
from tapnet.models.greedy import Greedy

import torch
import torch.nn as nn
import torch.nn.functional as F

class NewActor(nn.Module):
    def __init__(self, checkpoint_list, box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device) -> None:
        super().__init__()

        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim, corner_num)
        self.space_decoder = AttnModule(hidden_dim, heads=8, dropout=0, device=device, max_length=300, merge_flag=False)

        self.state_encoder = SpaceEncoder(ems_dim + box_dim, hidden_dim, corner_num=1)
        self.attn = SelfAttention(hidden_dim, heads=8)
        self.attn_embedding = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )


        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


        actor_list = []
        act_num = len(checkpoint_list)

        for ai in range(act_num):
            if checkpoint_list[ai] is None: continue
            
            if 'greedy' in checkpoint_list[ai]:
                actor = Greedy( pack_type='last', container_height=container_size[2], device=device).to(device)
            else:
                actor = Actor(box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device)
                actor = actor.to(device)

            if checkpoint_list is not None and 'greedy' not in checkpoint_list[ai]:
                print('Loading ', checkpoint_list[ai])
                state_dict = torch.load(checkpoint_list[ai], map_location=device)

                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model_dict = actor.state_dict()
                actor_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                actor.load_state_dict(actor_dict)
                
            actor.eval()

            actor_list.append(actor)
        
        self.actor_list = actor_list
        self.device = device

    def decode_actions(self, actions, batch_size, ems_num, box_state_num, box_states, ems):
        
        action_num = ems_num * box_state_num

        corner_ptr = actions % (action_num)
        
        ems_dim = ems.shape[2]
        box_dim = box_states.shape[2]
        
        ems_id = torch.div(corner_ptr, box_state_num, rounding_mode='floor')
        ems_id = ems_id.reshape(-1,1,1).expand(batch_size, 1, ems_dim)

        box_state_id = corner_ptr % box_state_num
        box_state_id = box_state_id.reshape(-1,1,1).expand(batch_size, 1, box_dim)

        new_box_states = box_states.gather(1, box_state_id)
        new_ems = ems.gather(1, ems_id)

        new_states = torch.cat([new_box_states, new_ems], dim=2)

        # corner_id = action // action_num
        # corner_id = corner_id % 4
        return new_states
        
    def forward(self, obs, state=None, info={} ):
        
        all_probs = []
        all_state = []

        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        ems_vecs = self.space_encoder(ems)
        for ems_vec in ems_vecs:
            container_vecs = self.space_decoder(ems_vec, ems_mask)

        all_new_states = []

        for actor in self.actor_list:
            probs, state = actor( obs, state=None, info={} )

            actions = probs.argmax(dim=1)
            new_states = self.decode_actions(actions, batch_size, ems_num, box_state_num, box_states, ems)

            all_new_states.append(new_states)
            
            all_probs.append(probs)
            
        # batch x actor_len x (ems_dim + box_dim)
        all_new_states = torch.cat(all_new_states, dim=1)
        state_vecs = self.state_encoder(all_new_states)[0]
        
        vecs = self.attn(container_vecs, container_vecs, state_vecs, None)
        attn_vecs = self.attn_embedding(vecs).reshape(batch_size, -1)

        attn_scores = F.softmax(attn_vecs, dim=1)

        # min_probs = 
        final_probs = 0
        for i in range(len(all_probs)):
            final_probs += attn_scores[:, i].reshape(-1, 1) * all_probs[i]
        
        # final_probs = F.softmax(final_probs, dim=1)

        return final_probs, state