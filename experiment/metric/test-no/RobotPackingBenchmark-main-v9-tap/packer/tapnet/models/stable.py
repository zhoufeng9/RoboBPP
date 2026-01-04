from tapnet.models.network import *

class StableNet(nn.Module):

    def __init__(self, box_dim, ems_dim, hidden_dim, prec_dim,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(StableNet, self).__init__()

        corner_num = 1
        self.corner_num = corner_num

        self.strategy = StrategyAttention( 'none', box_dim, ems_dim, hidden_dim, prec_dim, corner_num, device )

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.device = device
        self.box_dim = box_dim

    def forward(self, obs):
        
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        stable_vecs = self.strategy(box_states, prec_states, ems, ems_mask, ems_to_box_mask * 1, valid_mask)
        
        # b x 1 x  box_state_num
        valid_mask = valid_mask.unsqueeze(1).float().expand(batch_size, ems_num, box_state_num)

        # stable_score = stable_vecs * valid_mask
        stable_vecs = stable_vecs.reshape(batch_size, -1, ems_num, box_state_num)
        valid_score = stable_vecs.clone()
        for i in range(stable_vecs.shape[1]):
            valid_score[:,i,:,:] = valid_score[:,i,:,:] * valid_mask

        valid_score = valid_score.reshape(batch_size, -1)

        valid_score = F.sigmoid(valid_score)

        return valid_score

