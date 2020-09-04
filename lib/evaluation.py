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
        recalls = []
        mrrs = []

        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (history_S_id, pred_S_id) in tqdm(enumerate(eval_data_list)):
                history_S_id = history_S_id.to(self.device)
                pred_S_id = pred_S_id.to(self.device)

                input_length = history_S_id.size()[0]
                hidden = self.model.init_hidden()
                output = None
                for ei in range(input_length):
                    output, hidden = self.model(history_S_id[ei], hidden)
                print(torch.topk(output, self.topk, -1))

