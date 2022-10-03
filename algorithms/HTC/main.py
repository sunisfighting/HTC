
import torch
import numpy as np
from algorithms.HTC.MyNet import *
from algorithms.HTC.utils import *
from algorithms.network_alignment_model import NetworkAlignmentModel
from torch import optim
import torch.nn.functional as F
import time
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from input.dataset import Dataset
#from utils.graph_utils import load_gt
import utils.graph_utils as graph_utils
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

class HTC(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, groundtruth, args):
        super().__init__(source_dataset, target_dataset)
        self.src_data = source_dataset
        self.trg_data = target_dataset
        self.gt = groundtruth
        print('num of groundtruth: %d'%len(self.gt))
        self.args = args
        self.src_A, self.trg_A, self.src_feat, self.trg_feat = get_elements(source_dataset, target_dataset)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src_feat = self.src_feat.to(self.device)
        self.trg_feat = self.trg_feat.to(self.device)
        if self.args.first_run:
            src_goms = orca2gom(self.args.source_dataset, self.src_A)
            self.src_laps = torch.Tensor(gom2lap(src_goms)).to(self.device)
            trg_goms = orca2gom(self.args.target_dataset, self.trg_A)
            self.trg_laps = torch.Tensor(gom2lap(trg_goms)).to(self.device)
            torch.save(self.src_laps, args.source_dataset + '/src_laps.pt')
            torch.save(self.trg_laps, args.target_dataset + '/trg_laps.pt')
        else:
            print('loading orbit_laplacian matrices...')
            self.src_laps = torch.load(args.source_dataset + '/src_laps.pt').to(self.device)
            self.trg_laps = torch.load(args.target_dataset + '/trg_laps.pt').to(self.device)

        print('the shape of laps: ', self.src_laps.shape)
        self.num_node_s = self.src_feat.shape[0]
        self.num_node_t = self.trg_feat.shape[0]
        self.num_feat = self.src_feat.shape[1]
        print('attribute dimension: %d'%self.num_feat)
        self.num_hid1 = args.hid_dim
        self.num_hid2 = args.hid_dim
        self.utrain_epoch = args.num_utrn
        self.utrain_lr = args.ulr
        self.ftune_epoch = args.num_ftune
        self.ftune_lr = args.flr
        self.ftune_alpha = args.alpha
        self.k = args.k

    def align(self):

        myNet = MyNet(self.num_node_s, self.num_node_t, self.num_feat, self.num_hid1, self.num_hid2, self.args.p).to(self.device)

        myNet = self.unsupervised_train(myNet)

        myNet, count_max = self.trusted_refine(myNet)

        S_MyAlign = self.weighted_integration(myNet, count_max)

        return S_MyAlign

    def unsupervised_train(self, myNet):
        myNet.train()
        utrain_optimizer = optim.Adam(myNet.parameters(), lr=self.utrain_lr)
        rec_loss = Reconstruction_loss()
        loss_recorder = np.zeros((len(self.src_laps), self.utrain_epoch+1))
        for epoch in range(self.utrain_epoch):
            for i in range(len(self.src_laps)):
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
                src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
                trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
                trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
                loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
                loss_recorder[i, epoch] = loss_st
                print('epoch %d | loss_%d: %.4f' % (epoch, i, loss_st))
                utrain_optimizer.zero_grad()
                loss_st.backward()
                utrain_optimizer.step()
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
            src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
            trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
            trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
            loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
            loss_recorder[i, epoch+1] = loss_st
            print('epoch %d | loss_%d: %.4f' % (epoch+1, i, loss_st))
        return myNet

    def trusted_refine(self, myNet):
        if self.ftune_epoch>0:
            print('doing refinement')
            count_max = torch.zeros(len(self.src_laps))
            tune_flag = torch.ones(len(self.src_laps))
        else:
            count_max = torch.ones(len(self.src_laps))/len(self.src_laps)
            return myNet, count_max
        for epoch in range(self.ftune_epoch):
            if tune_flag.sum() == 0:
                print('done refinement')
                break
            for i in range(len(self.src_laps)):
                if tune_flag[i] == False:
                    print('epoch %d_%d: undo' % (epoch, i))
                    continue
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                csls = CSLS(src_output.detach(), trg_output.detach(), self.k)
                index_r = torch.argmax(csls, dim = 0)
                index_c = torch.argmax(csls, dim = 1)
                count = 0
                qs = torch.ones(len(self.src_laps[i])).to(self.device)
                qt = torch.ones(len(self.trg_laps[i])).to(self.device)
                for j in range(len(index_r)):
                    if j == index_c[index_r[j]]:
                        count += 1
                        qs[index_r[j]] *= self.ftune_alpha
                        qt[j] *= self.ftune_alpha
                qs = qs.reshape(-1, 1)
                qt = qt.reshape(-1, 1)
                if count > count_max[i]:
                    count_max[i] = count
                    self.src_laps[i] = (qs * (self.src_laps[i] * qs).t()).t()
                    self.trg_laps[i] = (qt * (self.trg_laps[i] * qt).t()).t()
                else:
                    tune_flag[i] = False
                print('epoch %d_%d: mutual closest pairs: %d' % (epoch, i, count))
        return myNet, count_max

    def weighted_integration(self, myNet, count_max):
        myNet.eval()
        total_count = sum(count_max)
        count_max = count_max / total_count
        score = torch.zeros((self.num_node_s, self.num_node_t)).to(self.device)
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            s = CSLS(src_output.detach(), trg_output.detach(), self.k)
            score += count_max[i] * s
        score = score.detach().cpu().numpy()
        return [score]
