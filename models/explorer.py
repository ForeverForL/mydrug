from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import utils
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np

from interfaces import Explorer



class PGLearner(Explorer, ABC):
    """ Reinforcement learning framework with policy gradient. This class is the base structure for the
        drugex v1 and v2 policy gradient-based  deep reinforcement learning models.
 
    Arguments:
 
        agent (models.Generator): The agent which generates the desired molecules
 
        env (utils.Env): The environment which provides the reward and judge
                                 if the generated molecule is valid and desired.
 
        prior: The auxiliary model which is defined differently in each methods.
    """
    def __init__(self, agent, env=None, mutate=None, crover=None, memory=None, mean_func='geometric', batch_size=128, epsilon=1e-3,
                 sigma=0.0, repeat=1, n_samples=-1, device=utils.dev, use_gpus=utils.devices):
        super().__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.replay = 10
        self.n_samples = 128  # * 8
        self.penalty = 0
        self.out = None
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func
 
    @abstractmethod
    def policy_gradient(self, smiles=None, seqs=None, memory=None):
        pass
 
    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, no_multifrag_smiles=True):
        best = 0
        last_save = 0
        log = open(self.out + '.log', 'w')
        for epoch in range(1000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            self.policy_gradient()
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True, no_multifrag_smiles=no_multifrag_smiles)
 
            desire = (scores.DESIRE).sum() / self.n_samples
            score = scores[self.env.getScorerKeys()].values.mean()
            valid = scores.VALID.mean()
 
            if best <= score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_save = epoch
 
            print("Epoch: %d average: %.4f valid: %.4f desired: %.4f" %
                  (epoch, score, valid, desire), file=log)
            for i, smile in enumerate(smiles):
                score = "\t".join(['%0.3f' % s for s in scores.values[i]])
                print('%s\t%s' % (score, smile), file=log)
            if epoch - last_save > 50:
                break
        for param_group in self.agent.optim.param_groups:
            param_group['lr'] *= (1 - 0.01)
        log.close()
 
 
class SmilesExplorerNoFrag(PGLearner):
    """ DrugEx algorithm (version 2.0)
 
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
 
    Arguments:
 
        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.
 
        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.
 
        mutate (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """
    def __init__(self, agent, env, mutate=None, crover=None, mean_func='geometric', memory=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, device=utils.dev, use_gpus=utils.devices):
        super(SmilesExplorerNoFrag, self).__init__(agent, env, mutate, crover, memory=memory, mean_func=mean_func, batch_size=batch_size, epsilon=epsilon, sigma=sigma, repeat=repeat, n_samples=n_samples, device=device, use_gpus=use_gpus)
        self.bestState = None
 
    def forward(self, crover=None, memory=None, epsilon=None):
        seqs = []
        #start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve(self.batchSize, epsilon=epsilon, crover=crover, mutate=self.mutate)
            seqs.append(seq)
        #t1 = time.time()
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s, is_tk = False) for s in seqs])
        # smiles = np.array(utils.canonicalize_list(smiles))
        ix = utils.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(self.device)]
        return smiles, seqs
   
    def policy_gradient(self, smiles=None, seqs=None, memory=None, progress=None):
        # function need to get smiles
        scores = self.env.getRewards(smiles, frags=None)
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batchSize * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]
        #t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(self.device))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
 
        # updating loss is done in rnn.py
        self.agent.PGLoss(loader, progress=progress)
        #t3 = time.time()
        #print(t1 - start, t2-t1, t3-t2)
 
    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, patience=50, no_multifrag_smiles=True):
        monitor.saveModel(self)
        self.bestState = deepcopy(self.agent.state_dict())
        max_desired_ratio = 0
        last_smiles = []
        last_scores = []
        last_save = -1
        ## add self.epoch

        for epoch in tqdm(range(epochs), desc='Fitting SMILES RNN explorer'):
            epoch += 1
            t0 = time.time()
            if epoch % 50 == 0 or epoch == 1: print('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < patience and self.memory is not None:
                smiles, seqs = self.forward(crover=None, memory=self.memory, epsilon=1e-1)
                self.policy_gradient(smiles, seqs, memory=self.memory, progress=monitor)
            else:
                smiles, seqs = self.forward(crover=self.crover, epsilon=self.epsilon)
                self.policy_gradient(smiles, seqs, progress=monitor)
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True, no_multifrag_smiles=True)
 
            desired_ratio = (scores.DESIRE).sum() / self.n_samples
            valid_ratio = scores.VALID.sum() / self.n_samples
            unique_ratio = len(set(smiles)) / len(smiles)

            if self.mean_func == 'arithmetic':
                mean_score = scores[self.env.getScorerKeys()].values.sum() / self.n_samples / len(self.env.getScorerKeys())
            else:
                mean_score = scores[self.env.getScorerKeys()].values.prod(axis=1) ** (1.0 / len(self.env.getScorerKeys()))
                mean_score = mean_score.sum() / self.n_samples

            t1 = time.time()
            print(f"Epoch: {epoch}  Score: {mean_score:.4f} Valid: {valid_ratio:.4f} Desire: {desired_ratio:.4f} Unique: {unique_ratio:.4f} Time: {t1-t0:.1f}s")

            smiles_scores = []
            smiles_scores_key = ['Smiles'] + list(scores.columns)
            for i, smile in enumerate(smiles):
                smiles_scores.append((smile, *scores.values[i]))
 
            scores['Smiles'] = smiles
            monitor.savePerformanceInfo(None, epoch, None, score=mean_score, valid_ratio=valid_ratio, desire_ratio=desired_ratio, unique_ratio=unique_ratio, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores_key)
            
            if max_desired_ratio < desired_ratio:
                monitor.saveModel(self)
                self.bestState = deepcopy(self.agent.state_dict())
                max_desired_ratio = desired_ratio
                last_save = epoch
                print(f"Model saved at epoch {epoch}")
 
            if epoch % patience == 0 and epoch != 0:
                # Every nth epoch reset the agent and the crover networks to the best state
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(['%.3f' % s for s in last_scores.drop(columns=['Smiles']).values[i]])
                    print('%s\t%s' % (score, smile))
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
                print('Resetting agent and crover to best state at epoch %d' % last_save)
            monitor.saveProgress(None, epoch, None, epochs)
            monitor.endStep(None, epoch)
    
            if epoch - last_save > patience: break
        
        print('End time reinforcement learning: %s \n' % time.strftime('%d-%m-%y %H:%M:%S', time.localtime()))
        monitor.close()