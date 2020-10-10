import torch

import babyai.utils as utils

from babyai.rl.utils import DictList


class MultiObssDictList(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
    def __len__(self):
        return len(next(iter(dict.values(self)))[0])
    
    def __getitem__(self, index):
        return DictList({key: [subvalue[index] for subvalue in value] for key, value in dict.items(self)})
    
    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


class MultiObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.obss_preprocessor = utils.ObssPreprocessor(model_name, obs_space, load_vocab_from)
        self.image_preproc = self.obss_preprocessor.image_preproc
        self.instr_preproc = self.obss_preprocessor.instr_preproc
        self.vocab         = self.obss_preprocessor.vocab
        self.obs_space     = self.obss_preprocessor.obs_space
    
    def __call__(self, obss, device=None):
        obs_ = MultiObssDictList()
        
        preprocessed_obss = [self.obss_preprocessor(obs, device=device) for obs in zip(*obss)]
        
        obs_.image = [preprocessed_obs.image for preprocessed_obs in preprocessed_obss]
        
        obs_.instr = [preprocessed_obs.instr for preprocessed_obs in preprocessed_obss]
        
        return obs_
