import lib
import numpy as np
import torch
from tqdm import tqdm

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            #for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                recall, mrr = lib.evaluate(logit, target, k=self.topk)

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr.item())
        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_recall, mean_mrr


    def eval_v1(self, eval_data_list):
        self.model.eval()
        next_interesting = []
        whole_day = []

        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (history_S_id, pred_S_id) in tqdm(enumerate(eval_data_list)):
                history_S_id = history_S_id.to(self.device)
                #pred_S_id = pred_S_id.to(self.device)

                input_length = history_S_id.size()[0]
                hidden = self.model.init_hidden()
                output = None
                end = min(3, input_length)
                start = max(0, end - 1)
    
                for ei in range(end):
                    output, hidden = self.model(history_S_id[ei], hidden)

                if output is None:
                    continue
                else:
                    _, pred = torch.topk(output[0], self.topk, -1)
                    pred = pred.tolist()
                    #print(pred_S_id, pred)
                    metrics_map = ['HR', 'HR@', 'MRR', 'NDCG']
                    out = lib.metrics(pred_S_id, pred, metrics_map)
                    #print(out)
                    next_interesting.append(out)

                    metrics_map = ['P&R', 'MAP']
                    out =lib. metrics(pred_S_id, pred, metrics_map)
                    #print(out[0], [out[1]])
                    whole_day.append(out[0] + [out[1]])

        interesting_metric = np.array(next_interesting)
        day_metric = np.array(whole_day)
        if len(interesting_metric) == 0:
            HR_interesting, HR_at_interesting, MRR_interesting, NDCG_interesting = 0, 0, 0, 0
        else:
            HR_interesting, HR_at_interesting, MRR_interesting, NDCG_interesting = np.mean(interesting_metric, axis=0).tolist()
        
        if len(day_metric) > 0:
            Precison, Recall, F1, MAP = np.mean(day_metric, axis=0).tolist()
        else:
            Precison, Recall, F1, MAP = 0, 0, 0, 0

        print(f'HR:{HR_interesting:.4f}\tHR@10:{HR_at_interesting:.4f}\tMRR:{MRR_interesting:.4f}\tNDCG:{NDCG_interesting:.4f}')
        print(f'P:{Precison:.4f}\tR:{Recall:.4f}\tF1:{F1:.4f}\tMAP:{MAP:.4f}')

