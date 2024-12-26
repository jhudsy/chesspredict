from keras import config

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras import Model

import tensorflow as tf

import chess
import chess.pgn


import numpy as np

print(config.backend())

########################################
def get_ratings(game):

    return int(game.headers['WhiteElo']),int(game.headers['BlackElo'])

def rating_to_output(rating):
    ret = np.zeros(48)
    r = int((rating-600)/50)
    if r>47:
        r = 47
    if r<0:
        r = 0
    ret[r] = 1
    return ret
########################################

def get_game_tensors(game_pgn,num_tensors):

    gt = np.zeros((num_tensors,134))

    moves = []
    evals = []

    for m in game_pgn.mainline():
        moves.append(m.san())
        evals.append(m.eval().white() if m.turn() == chess.WHITE else m.eval().black())

    #let our t vector be a 1D array of 133 elements. The first 128 element represent the board before the move is made and after the move is made. The 129th element is the evaluation of the move before it is made and the 130th element is the evaluation of the move after it is made. If it is mate in X moves before the move is made, the 131st element is 1 and 0 otherwise. If it is mate in X moves after the move is made, the 132nd element is 1 and 0 otherwise. The 133rd element is 1 if it is white moved and -1 if it is black making the move.

    board = chess.Board()

    for m in range(0,min(40,len(moves)-1)):
        t = np.zeros(134)

        for i in range(64): #original board position
            if board.piece_at(i) is not None:
                t[i] = board.piece_at(i).piece_type * (1 if board.piece_at(i).color == chess.WHITE else -1)
    
        board.push_san(moves[m])
        for i in range(64):
            if board.piece_at(i) is not None:
                t[i+64] = board.piece_at(i).piece_type * (1 if board.piece_at(i).color == chess.WHITE else -1)
    
        #evals is either a number, or starts with a #. If it starts with a #, it is a mate in X moves
        if evals[m].startswith('#'):
            t[129] = float(evals[m][1:])
            t[130] = 1
        else:
            t[129] = float(evals[m]/100)
            t[130] = 0
    
        if evals[m+1].startswith('#'):
            t[131] = float(evals[m+1][1:])
            t[132] = 1
        else:
            t[131] = float(evals[m+1]/100)
            t[132] = 0

        t[133] = -1 if board.turn == chess.WHITE else 1

        gt[m] = t

    return gt

########################################

#our input is a 40x66 matrix. We will use 40 time steps and 66 features. The first 64 features are an 8x8 image and the last two features are a (0,1) and a float. The image itself is a chessboard and consists of elements taken from the categorical set {0..12} where 0 is an empty square and 1..12 are the 6 types of pieces for each player. The (0,1) feature is a flag that indicates which player is to move next and the float is the value of the position as evaluated by Stockfish.

input = Input(shape=(40, 134))
x = LSTM(128, return_sequences=True)(input)
x = LSTM(128)(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#The output is a pair of 2 vectors of length 48, using a one-hot encoding. Each vector is the player's rating. The first vector is for white, and the seocnd is for black. The ratings are in the range 0..47, where 0 is the lowest rating and 47 is the highest rating. The ratings are integers and are distributed uniformly in the range 0..47.

output1 = Dense(48, activation='softmax')(x)
output2 = Dense(48, activation='softmax')(x)

model = keras.Model(inputs=input, outputs=[output1, output2])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

########################################

#load the data. Each file in TRAININGDIR contains a single game as a string (gamestring) which can be parsed with the get_game_tensors(gamestring) and rating_to_output(get_ratings(gamestring)[0]),rating_to_output(get_ratings(gamestring)[1]). We also have a VALIDATIONDIR with the same format.

TRAININGDIR = 'training/'
VALIDATIONDIR = 'validation/'

#######################################
def process_file(file_path):
    game = chess.pgn.read_game(file_path)
    game_string = str(game)
    gt = get_game_tensors(game_string,40)
    white_rating, black_rating = get_ratings(game_string)
    return gt, (white_rating, black_rating)


#we will have a batch of 32 games at a time for training

batch_size = 32

#we will train for 100 epochs or when validation loss starts to increase

epochs = 100

#prepare the training data using a tf dataset
tds = tf.data.Dataset.list_files(str('data/training/*'))
tds = tds.map(process_file, num_parallel_calls=tf.data.AUTOTUNE)
tds.batch(32)
tds.prefetch(buffer_size=tf.data.AUTOTUNE)


#we will save the model with the best validation loss and use 10% of the training data for validation

model.fit(tds, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)])

model.save('model.h5')


