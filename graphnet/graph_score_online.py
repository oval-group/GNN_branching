import torch
from graphnet.graph_conv import GraphNet
#from training_utils import hinge_rank_loss
import time


class GraphChoice:

    def __init__(self, init_mask, model_name, lr=1e-4, wd=1e-4, linear=False):
        model =  GraphNet(2, 64)

        model.load_state_dict(torch.load(model_name))
        model.eval()
        self.model = model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = wd)
        trans_len = []
        temp = 0
        for i in init_mask:
            temp += len(i)
            trans_len.append(temp)
        self.trans_len = torch.tensor(trans_len)

    def decision(self, lower_bounds_all, upper_bounds_all, dual_vars, primal_input, primals,layers, mask):
        mask = [(i==-1).float() for i in mask]
        mask_1d = torch.cat([i for i in mask], 0)
        self.mask_1d = mask_1d.unsqueeze(0)
        start = time.time()
        lower_bounds_all = [i.cuda() for i in lower_bounds_all]
        upper_bounds_all = [i.cuda() for i in upper_bounds_all]
        dual_vars = [i.cuda() for i in dual_vars]
        primal_input = primal_input.cuda()
        primals = [torch.tensor(i).cuda() for i in primals]

        self.scores = self.model(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_input, layers, self.mask_1d)
        end = time.time()
        print(f'graph requires: {end-start}')
        scores = self.scores[0]
            

        
        self.gnn_score, choice = torch.max(scores,0)
        idx = (self.mask_1d[0].nonzero()[choice]).item()
        dec_lay =  ((self.trans_len>idx).nonzero()[0]).item()
        if dec_lay == 0:
            dec_idx = idx
        else:
            dec_idx = (idx - self.trans_len[dec_lay-1]).item()

        #if dec_lay ==0:
        #    import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()

        #print('final decision: ', [dec_lay, dec_idx])
        #end = time.time()
        #print(f'graph requires: {end-start}')

        # compute kw dom loss under the assumption kw is a better decision

        return [dec_lay, dec_idx]


    def online_learning(self, kw_decision, improvement):
        if kw_decision[0] == 0:
            partial_len = 0
        else:
            partial_len = self.trans_len[kw_decision[0]-1]
        print('updating the trained model')
        partial_len = partial_len + kw_decision[1]
        kw_index = len(self.mask_1d[0][:partial_len].nonzero())
        kw_score = self.scores[0][kw_index]
        #assert kw_score == self.new_score[kw_decision[0]][kw_decision[1]]
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gnn_score - kw_score + improvement
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def del_score(self):
        del self.scores
        del self.gnn_score
        del self.mask_1d
        #del self.new_score







