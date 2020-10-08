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


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_
