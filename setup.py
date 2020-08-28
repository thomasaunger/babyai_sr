from setuptools import setup, find_packages

setup(name     = "babyai_sr",
      version  = "1.0",
      license  = "BSD-3-Clause",
      keywords = "reinforcement learning, language emergence, multi-agent systems, language grounding, unsupervised semantics, deep learning, transfer learning",
      packages = find_packages(),
      install_requires = [
                          "matplotlib>=3.3.1"
                         ]
     )
