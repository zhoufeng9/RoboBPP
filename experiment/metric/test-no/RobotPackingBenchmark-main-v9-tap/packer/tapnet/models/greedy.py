import torch
import torch.nn as nn
import torch.nn.functional as F
from tapnet.models.encoder import obs_to_tensor

class Greedy(nn.Module):
    def __init__(self, pack_type, container_height, device, *kwargs) -> None:
        super(Greedy, self).__init__()
        self.greedy = nn.Linear(1,1)
        self.device = device
        self.pack_type = pack_type
        self.container_height = container_height
    
    def forward(self, obs, state=None, info={} ):
        # comapre max reawrd: size / max_h

        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        # pos | size
        # ems: batch x ems_num x 6
        # box_states: batch x (rot_num * axis_num * box_num) x 3
        
        prec_mask = access_mask.clone() * 1

        if False:
            max_prec = prec_states.sum(dim=3).sum(dim=1) # batch x box_num
            box_num = max_prec.shape[-1]
            state_num = box_state_num / box_num
            
            max_prec[ ~valid_mask[:, :box_num] ] = 0
            max_prec[ ~access_mask[:, :box_num] ] = 0
            # NOTE bug in mask

            max_mask = max_prec == max_prec.max(dim=1)[0].unsqueeze(1)
            # max_mask = max_prec.argmax(dim=1)

            prec_mask = access_mask.clone() * 0
            for s in range(int(state_num)):
                prec_mask[:, s*box_num:(s+1)*box_num][ max_mask ] = 1

        
        # batch x 1 x (rot_num * axis_num * box_num)
        box_z = box_states[:, :, 2].unsqueeze(1)

        # batch x 1 x (rot_num * axis_num * box_num)
        box_size = box_states[:, :, 0] * box_states[:, :, 1] * box_states[:, :, 2]
        box_size = box_size.unsqueeze(1)

        # batch x ems_num x 1
        ems_z = ems[:, :, 2:3]

        ems_h = ems[:, :, 5]
        
        # ems_id = ems[:, :, 6]

        # batch

        # batch x ems_num x box_z
        ems_to_box_height = ems_z + box_z

        if self.pack_type == 'all' or self.pack_type == 'last':
            max_h = torch.max(ems_h, dim=1)[0].unsqueeze(1).unsqueeze(1)
            max_h = max_h.expand(batch_size, ems_num, box_state_num)

            higher_mask = ems_to_box_height < max_h
            if higher_mask.max() == True:
                ems_to_box_height[higher_mask] = max_h[higher_mask]

        # elif self.pack_type == 'last':
        # ems_to_box_height = self.container_height

        # reward
        # sh_rate = box_size / self.container_height
        sh_rate = -ems_to_box_height
        # add mask for valid ems-box pair
        height_score = sh_rate + ems_to_box_mask.float().log() + (prec_mask * valid_mask * access_mask).unsqueeze(1).float().log()

        height_score = height_score.reshape(batch_size, -1)
        
        # batch x ...
        probs = F.softmax(height_score, dim=1)

        # lower is better

        low_ems_to_box_height = (1 - ems_to_box_height) #+ ems_to_box_mask.float().log() + (prec_mask * valid_mask * access_mask).unsqueeze(1).float().log()
        low_ems_to_box_height = low_ems_to_box_height.reshape(batch_size, -1)
        # low_height_probs = F.softmax(low_ems_to_box_height, dim=1)
        low_height_probs = low_ems_to_box_height
        new_probs = probs * low_height_probs
        probs = F.softmax(new_probs, dim=1)
        
        if self.training == False:
            prob_max = probs.max(dim=1, keepdim=True)[0]
            probs[ probs != prob_max ] = 0
            probs[ probs == prob_max ] = 1
            probs /= probs.sum(dim=1, keepdim=True)
        
        return probs, state
        

class Critic(nn.Module):
    def __init__(self, device, *kwargs) -> None:
        super(Critic, self).__init__()
        self.greedy = nn.Linear(1,1)
        self.device = device

    def forward(self, obs, act=None, info={} ):
        batch_size = len(obs.box_num)
        output_value = torch.ones( batch_size, 1 ).to(self.device)
        return output_value
        
    
class Actor(nn.Module):
    def __init__(self, device, *kwargs) -> None:
        super(Actor, self).__init__()
        self.actor = Greedy(device)
    
    def forward(self, obs, state=None, info={} ):
        probs, state = self.actor(obs, state=None, info={})
        return probs, state