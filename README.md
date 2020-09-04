# BabyAI S/R

This is a package that uses BabyAI in a sender/receiver setup.

## Installation

First, install [this BabyAI fork](https://github.com/thomasaunger/babyai) and the corresponding [MiniGrid fork](https://github.com/thomasaunger/gym-minigrid), which ease compatibility. (If you’re feeling lucky, you can instead try installing the [original BabyAI repo](https://github.com/mila-iqia/babyai) (and the corresponding [original MiniGrid repo](https://github.com/maximecb/gym-minigrid)).)

Then, clone this repository and install it with `pip3`:

```
git clone https://github.com/thomasaunger/babyai_sr.git
cd babyai_sr
pip3 install --editable .
```

In order to use the plotting and visualization scripts as is, you’ll also need to install [LaTeX](https://www.latex-project.org).

## Usage

This package is organized similarly to the original BabyAI repo. Models can be trained using `scripts/train_rl.py`.

The resulting models can be tested using `scripts/test_rl.py`, which also saves the episode data to an NPY file, which in turn can be visualized using the `vis` scripts.

The learning curves as described by the log files can be plotted using `scripts/plot_curves.py`.
