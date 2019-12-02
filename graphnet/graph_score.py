import torch
from graphnet.graph_conv import GraphNet
import time


class GraphChoice:

    def __init__(self, init_mask, model_name, linear=False):
        model =  GraphNet(2, 64)

        model.load_state_dict(torch.load(model_name))
        model.eval()
        self.model = model.cuda()
        trans_len = []
        temp = 0
        for i in init_mask:
            temp += len(i)
            trans_len.append(temp)
        self.trans_len = torch.tensor(trans_len)

    def decision(self, lower_bounds_all, upper_bounds_all, dual_vars, primal_input, primals,layers, mask):
        mask = [(i==-1).float() for i in mask]
        mask_1d = torch.cat([i for i in mask], 0)
        mask_1d = mask_1d.unsqueeze(0)
        start = time.time()
        lower_bounds_all = [i.cuda() for i in lower_bounds_all]
        upper_bounds_all = [i.cuda() for i in upper_bounds_all]
        dual_vars = [i.cuda() for i in dual_vars]
        primal_input = primal_input.cuda()
        primals = [torch.tensor(i).cuda() for i in primals]

        with torch.no_grad():
            #scores = self.model(lower_bounds_all, upper_bounds_all, dual_vars, primal_input, primals,layers, mask_1d)
            scores = self.model(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_input, layers, mask_1d)
        end = time.time()
        print(f'graph requires: {end-start}')
        scores = scores[0]


        
        _, choice = torch.max(scores,0)
        idx = (mask_1d[0].nonzero()[choice]).item()
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
        return [dec_lay, dec_idx]






