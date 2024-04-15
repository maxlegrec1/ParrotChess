
# ParrotChess : Mimicking Human Play in Chess

Depth-0 neural network based chess engine with SOTA performance on mimicking human moves.




![Logo](https://media1.tenor.com/m/Yb0fYsfyksQAAAAd/chesscom-chess.gif)


Based on Leela Chess Zero newest transformer architecture, our models learn to play the human moves in each position (policy-only training), similarly to Maia.


## Performances 

| Model             | Policy Accuracy |
| ---               | ---             |
| Maia              | 53 %            |
| ParrotChess       | 58 %            |


In contrary to Maia and DeepMind, our model is given additional information : ```Player Elo, Time Control``` that helps him to predict the move played.

This means that in contrary to Maia, which uses a range of models (one for each 100 elo bin), our model can adapt to any elo


Insert training losses, metris, cool graphs here.
Graphs should include accuracy per elo, accuracy per move number.

![Elo Distribution](/assets/Elo_Distribution.png)
![Accuracy per Elo](/assets/Accuracy_Per_Elo.png)
![Accuracy per Move Ply](/assets/Accuracy_Per_ply.png)

## Installation

Install ParrotChess with Git

```bash
  git clone https://github.com/maxlegrec1/ParrotChess.git
  cd ParrotChess/
```


    
## Requirements

Install the required python modules with pip.

```bash
pip install -r requirements.txt
```
    
## Train a model

The models are trained through supervised learning with games extracted from the [Lichess Database](https://database.lichess.org/) . Given a position extracted from a human game, the models learn to play the move played by the human.

### Download a dataset 

Using the [previous link](https://database.lichess.org/) to download pgn files, and put them inside the same folder. This folder will be ``` path_pgn``` that you have to enter in the config.

### Chose training hyperparameters

Edit the config file ```Refactored/config/default_config.py``` 
 to change the batch size, learning rates,the path of the dataset, and other parameters. Both absolute and relative paths should work.


Make sure that ```Resume_id``` is ```None``` if you are starting a new training.


### Train

run ```python Refactored/main.py``` to launch the training.

Connect to weights and biases through the terminal if needed.


## Data

This section is unfinished, and will cover all the data transformation.

This section should in term, cover the rejection sampling algorithm, shuffle by transposing, Parallel Generation.
### Input Shape

The networks have two inputs ```Input1``` and ```Input2``` of respective shape ```(_,8,8,102)``` and ```(_,8,8,2)```.

The first 96 planes of ```Input1``` correspond to the current and last 7 positions. The remaining 6 planes correspond to Castling Rights (4), En Passant Right (1), and Color (1).

The 2 planes of ```Input2``` correspond to the time control of the game in minutes, normalized by 300, and the elo of the player, normalized between 0 and 1. 

## Losses and Metrics

Click [here](https://wandb.ai/maxlegrec/owt) to see the training losses and other metrics.

## Available models

### ParrotChess
The first model that I trained over 2M steps with batch size 256 on games of every elo.
The architecture is a Transformer Encoder with 15 layers, 1024 embedding size, 1536 dense feed-forward, for a total of 140M parameters. 

It achieves 57.5% of policy accuracy overall.

Download the model [here](https://wandb.ai/maxlegrec/owt/runs/9puxko1e/files?nw=nwusermaxlegrec)

### ParrotChess Pro
The second model that I have trained, on games of elite players (2400+ on lichess).
The architecture is a Mixture of Experts Transformer Encoder of 10 layers, 512 embedding size, 16*736 dense feed-forward, for a total of 100M parameters. 

Download the model [here](https://wandb.ai/maxlegrec/owt/runs/t4sltaps/files?nw=nwusermaxlegrec)

It achieves over 62% accuracy on the move of pro players. Way better than [Leela Chess Zero](https://lczero.org/)'s BT4 network at depth 0.
## Acknowledgements

 - [Maia Chess](https://maiachess.com/)
 - [Leela Chess Zero](https://lczero.org/)
 - [Google Deepmind Latest publications](https://arxiv.org/abs/2402.04494)


## Authors

- [@maxlegrec1](https://www.github.com/maxlegrec1)

- [@IAntoineV](https://www.github.com/IAntoineV)

