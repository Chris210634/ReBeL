# EC700 Reinforcement Learning Project

In this project I reproduce the ReBeL algorithm found here: https://github.com/facebookresearch/rebel

I use the ReBeL algorithm on a game called Poker Dice (A very simplified version of Poker).

The source code for the Poker Dice game can be found in csrc/poker_dice.

## Installation and Setup

I tested on the BU SCC. 

First, load the appropriate modules.
```
module load gcc/10.2.0 cuda miniconda
```

Then, run the following to create a conda environment and install the appropriate packages.
```
git clone https://github.com/Chris210634/ReBeL
cd ReBeL
mkdir .conda
conda create --prefix .conda/rebel
conda activate .conda/rebel
pip install -r requirements.txt   # This might not work
conda install cmake
git submodule update --init
```

Compile the C++ part:
```
make
```

Add current dir to Python path:
```
export PYTHONPATH=$PYTHONPATH:$PWD
```

It may take a few tries to get the Python packages to work well with each other.
In particular, the version of pybind, pytorch, gcc, and python must be compatible.

I used Python 3.7.9, GCC 10.2.0, PyTorch 1.7.0.

Please see ```requirements.txt``` for the full list of python packages and versions in my enviornment.

## Running the Program

You need 1 V100 GPU.

### Training

To train the value network, run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python run_poker.py
```
You can change hyperparameters in ```conf/c02_selfplay/poker_ap.yaml```.
You will see the replay buffer being filled. Once a certain number of training examples is reached, the value network is trained one epoch.
Most of the time is spent generating the training examples.
I found that training the value network one epoch is sufficient to get reasonable results.

checkpoints are stored in the ```outputs``` folder.

### Evaluation
To evaluate the value network policy, edit the ```bin_path``` variable in ```eval_net.py``` to point to the value net checkpoint from the previous step. run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/eval_net.py 
```

This script calculates the exploitability of the value net strategy by averaging over all possible games (total 216).

To run full-game CFR, run the following:

```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/full-cfr.py
```

This script runs Full game CFR on all 216 possible games and reports the average exploitability. You can change the number of CFR iterations by editing the code in ```full-cfr.py```.

### Demo

To watch the game between ReBeL and full game CFR, run the following (Keep hitting enter):
```
TORCH_USE_RTLD_GLOBAL=YES python cfvpy/play_poker_dice.py
```

This script randomly generates games. You can hit enter to see the computer play the next move. A sample output is included in the next section.

## Results

### Demo Results

Example output from game between ReBeL and full game CFR:

```
NEW GAME
Neural Net is Player 0 and Full-game CFR is Player 1

public hand: 1 6 2
private hand Player 0: 6 1
private hand Player 1: 4 1
State: (pid=0,pub-hand=66,last=2,event=raise)
Action Probabilities (Fold, Call, Raise): 2.5283e-06 1.15245e-05 0.999986
Player: 0 Action: raise
State: (pid=1,pub-hand=66,last=3,event=raise)
Action Probabilities (Fold, Call, Raise): 0.947801 0.00026067 0.0519379
Player: 1 Action: fold
Winner is Player 0 Bet was 2
```

In the above game, Player 0 is ReBeL and Player 1 is full game CFR.

The three public dice rolled were {1 6 2}. Player 0 rolled {6 1} and Player 1 rolled {4 1}.

Player 0 went first. You can see the player calculated a high probability for raising, probably becuase it had two pair (a pair of 1s and a pair of 6s).

Player 0 samples from that probability distribution and chooses to raise.

Player 1 then calculates its probabability distribution, which has about 95% chance of folding. Player 1 samples from this probability distribution and indeed chooses to fold.

Player 1 loses 2 points because it folded and the bid was 2.

### Exploitability Results


![convergence](https://user-images.githubusercontent.com/10382186/117016257-86c9c300-acc0-11eb-9a15-4b42daad0203.PNG)


