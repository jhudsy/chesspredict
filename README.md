# Guess the Elo

This project trains a neural network to guess the elo of chess games. Training took place on a subset of the lichess open database.

## Installation

`pip3 install requirements.txt` should give you everything you need.

## Workflow

The typical workflow is as follows.

### File Preparation

This step transforms a bunch of games into files suitable for machine learning.

  1. Download one or more games from the [lichess games database](https://database.lichess.org/). These games should be from after April 2017 to include clock information.
  2. Run `python3 -m ml.data_prep read_file <filename> <target directory>`. This will create a bunch of hdf5 files (e.g., `blitz.hdf5`) as described by the `file_dict` in `config.py`. You will typically have separate files for different time controls. You can run this multiple times to process multiple files, the hdf5 files will be appended to with additional data.
  3. Run `python3 -m ml.data_prep split <filename> <path> --training <train> --validation <val> --test <test>`. This will take one of the files created by the previous step and split it further into a bunch of `bin_<X>.hdf` files for training data, `validation.hdf5` and `test.hdf5` files for validation and test data respectively, according to the `<train>/<val>/<test>` split (e.g., 0.8/0.1/0.1) specified. You can also use a `--min_rating` and `--max_rating` tag. Anything below/above these ratings will be put into the lowest/highest bin. *You should ensure there are at least 128 elements in each training bin (or modify the program to reduce the cache size for the training generator, see below).

### Learning

The neural network architecture is specified in `learning.py` within the `make_model` function. You can either use/adapt this, or search for an ideal model using the following commands. The expectation is that a separate model will be trained for each of the files in `file_dict` seperately as evaluating elo for a blitz game is very different to doing so for a classical game. More or fewer files can be created by changing the `file_dict` itself to test this hypothesis.

   - `python3 -m ml.learning train <training path> <validation file> <testing file> <model file>`. The paths/filenames are as per the third step above. Additional parameters are `--batch_size`, `--shuffle` (True of False) and `--cache_size` (making it bigger will use more memory but will marginally speed things up). The latter setting should be modified if you have few entries in any bin. The model file is the name of the file where the learned model will be saved.
   - Instead of `train` you can use `tune` which will take a neural network outline (see `model_builder` function) and try find the best neural network size.
   - The `evaluate` action will run the network on the testing dataset.
   - Running `python3 -m ml.learning predict <model file> <pgn file>` will return two numbers, the guessed elo for white, and the guessed elo for black.

## Web server

A flask app is also included as part of the package. Installation instructions for this are not given (both because I'm lazy and also because I always struggle to remember what I've done to set up a web server).

## The network and underlying system, suggestions for improvements.

The input to the system is a 40x135 element tensor representing all of one color's moves within a game, together with some additional elements. Each of the 40 elements represents a single move by that color (i.e., we consider only the first 40 moves of the game. This parameter is tunable via `NUM_MOVES` in `config.py`). The 136 elements of the tensor are as follows.
  - The first 64 elements are the position of each piece before the move on the chessboard recast as a 1-D array. Each piece is a different integer, and the player and opponent  colors have different values too (irrespective of whether the player is white or black). _Investigating whether flipping the board for the different colors may aid the model via `board.mirror()` is left for the future_.
  - The second 64 elements are the positions of each piece after the move is made, as per the above.
  - Element 128 (as we're 0 indexed) is the current move number divided by 2, i.e., the number of full moves so far.
  - Element 129 is the current player's time before the move is made.
  - Element 130 is the current player's time after the move is made.
  - Element 131 is the opponent's time.
  - Element 132 is either the centipawn score or X where X is how many moves till mate, from the current player's point of view, before the move is made.
  - Element 133 is 1 if the position is a mate in X, 0 otherwise, before the move is made.
  - Element 134 is either the centipawn score or X where X is how many moves till mate, from the current player's point of view, after the move is made.
  - Element 135 is 1 if the position is a mate in X, 0 otherwise, after the move is made.
  - If the game is a mate then elmenent 135 is a 1 and element 134 is a 0.

The best architecture I've found (so far) has a Time Distributed layer containing a dense layer of 80-100 neurons with a RELU activation function. The next layer is a 40 unit LSTM followed by another LSTM of roughly 32 units, with another +-60 neuron RELU activated dense layer and a final 1 neuron output layer to give the elo. 

At the moment, the system has a MAE of around 200, but performs much better on the handful of games I've tested it with manually.

I have only evaluated the system on around 200000 blitz games (5+1 and 5+0). Additional training (on more games and time controls) and more of a hyperparameter search, as well as e.g., mirroring the board as noted above will be interesting. More detailed analysis of the systems behaviour on different input games is another TODO.



