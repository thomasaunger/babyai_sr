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
    def __init__(self, model_names, obs_spaces=None, load_vocabs_from=None):
        if obs_spaces is None:
            obs_spaces = [None]*len(model_names)
        
        if load_vocabs_from is None:
            load_vocabs_from = [None]*len(model_names)
        
        self.obss_preprocessors = [utils.ObssPreprocessor(model_name, obs_spaces[m], load_vocabs_from[m]) for m, model_name in enumerate(model_names)]
        self.image_preprocs = [obss_preprocessor.image_preproc for obss_preprocessor in self.obss_preprocessors]
        self.instr_preprocs = [obss_preprocessor.instr_preproc for obss_preprocessor in self.obss_preprocessors]
        self.vocabs         = [obss_preprocessor.vocab         for obss_preprocessor in self.obss_preprocessors]
        self.obs_spaces     = [obss_preprocessor.obs_space     for obss_preprocessor in self.obss_preprocessors]
    
    def __call__(self, obss, device=None):
        obs_ = MultiObssDictList()
        
        preprocessed_obss = [self.obss_preprocessors[m](obs, device=device) for m, obs in enumerate(zip(*obss))]
        
        obs_.image = [preprocessed_obs.image for preprocessed_obs in preprocessed_obss]
        
        obs_.instr = [preprocessed_obs.instr for preprocessed_obs in preprocessed_obss]
        
        return obs_
