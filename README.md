
# ParrotChess : Mimicking Human Play in Chess

Depth-0 neural network based chess engine with SOTA performance on mimicking human moves.




![Logo](https://media1.tenor.com/m/Yb0fYsfyksQAAAAd/chesscom-chess.gif)


Based on Leela Chess Zero newest transformer architecture, our models learn to play the human moves in each position (policy-only training), similarly to Maia.


## Performances 

| Model             | Policy Accuracy |
| ---               | ---             |
| Maia              | 53 %            |
| DeepMind's GM AI  | 57 %            |
| ParrotChess       | 58 %            |


In contrary to Maia and DeepMind, our model is given additional information : ```Player Elo, Time Control``` that helps him to predict the move played.

This means that in contrary to Maia, which uses a range of models (one for each 100 elo bin), our model can adapt to any elo


Insert training losses, metris, cool graphs here.
Graphs should include accuracy per elo, accuracy per move number.

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

Using the [previous link](https://database.lichess.org/) to download a dataset (pgn file) and put it in the ParrotChess directory ```ParrotChess/```.

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
## Metrics

Click [here](https://wandb.ai/maxlegrec/owt) to see the training losses and other metrics.
## Acknowledgements

 - [Maia Chess](https://maiachess.com/)
 - [Leela Chess Zero](https://lczero.org/)
 - [Google Deepmind Latest publications](https://arxiv.org/abs/2402.04494)


## Authors

- [@maxlegrec1](https://www.github.com/maxlegrec1)

- [@IAntoineV](https://www.github.com/IAntoineV)

