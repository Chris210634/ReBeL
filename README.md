# EC700 Reinforcement Learning Project

In this project I reproduce the ReBeL algorithm found here: https://github.com/facebookresearch/rebel

I use the ReBeL algorithm on a game called Poker Dice (A very simplified version of Poker).

## Installation and Setup

I tested on the SCC.

## Running the Program

To train the value network, run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python run_poker.py
```

To evaluate the value network policy, run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/eval_net.py 
```

To run full-game CFR, run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/full-cfr.py
```

To watch the game between ReBeL and full game CFR, run the following:
```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/play_poker_dice.py
```
