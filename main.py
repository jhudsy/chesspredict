import keras

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import CategoryEncoding

import os

import tensorflow as tf

import chess
import chess.pgn


import numpy as np

from tensorflow.python.client import device_lib

print("Keras backend:",keras.config.backend())
print(device_lib.list_local_devices())

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
        if m.eval() is None: #should only happen on a mate
            evals.append(None)
        else:
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

        #handle the case of a mate
        if evals[m] is None:
            t[129] = 0
            t[130] = 1
        if evals[m+1] is None:
            t[131] = 0
            t[132] = 1

        if evals[m] is not None and evals[m].is_mate():
            t[129] = float(evals[m].mate())
            t[130] = 1
        elif evals[m] is not None:
            t[129] = float(evals[m].score()/100)
            t[130] = 0
    
        if evals[m+1] is not None and evals[m+1].is_mate():
            t[131] = float(evals[m+1].mate())
            t[132] = 1
        elif evals[m+1] is not None:
            t[131] = float(evals[m+1].score()/100)
            t[132] = 0

        t[133] = -1 if board.turn == chess.WHITE else 1

        gt[m] = t

    return gt

########################################

#load the data. Each file in TRAININGDIR contains a single game as a string (gamestring) which can be parsed with the get_game_tensors(gamestring) and rating_to_output(get_ratings(gamestring)[0]),rating_to_output(get_ratings(gamestring)[1]). We also have a VALIDATIONDIR with the same format.

TRAININGDIR = 'data/training/'
VALIDATIONDIR = 'data/validation/'

class GameSequence(keras.utils.Sequence):

    def __init__(self, pgn_dir, batch_size=32, shuffle=True,**kwargs):
        self.cache = {}
        
        super().__init__(**kwargs)
        #self.pgn_files should be the full path to all files in pgn_dir
        self.pgn_files=[pgn_dir+f for f in os.listdir(pgn_dir) if os.path.isfile(os.path.join(pgn_dir, f))]
        
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.pgn_files)
    
    def __len__(self):
        return len(self.pgn_files) // self.batch_size
    
    def __getitem__(self, idx):
        
        batch_files = self.pgn_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y1,y2 = [], [], []
        for file in batch_files:
            if file not in self.cache:
                with open(file) as f:
                    pgn = chess.pgn.read_game(f)

                    self.cache[file] = get_game_tensors(pgn,40), (get_ratings(pgn)[0],get_ratings(pgn)[1])
                    #self.cache[file] = get_game_tensors(pgn,40), (rating_to_output(get_ratings(pgn)[0]),rating_to_output(get_ratings(pgn)[1])) #use if using one-hot encoding

            
            X.append(self.cache[file][0])
            y1.append(self.cache[file][1][0])
            y2.append(self.cache[file][1][1])
                         
        X = np.array(X)
        y1 = np.array(y1)
        y2 = np.array(y2)

        #shape is (32, 40, 134) (32, 2, 48) for X and y respectively with a batch size of 32
        
        return X, (y1,y2)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pgn_files)


########################################
#we will have a batch of 32 games at a time for training
batch_size = 16

#we will train for 100 epochs or when validation loss starts to increase
epochs = 100

#our input is a 40x66 matrix. We will use 40 time steps and 66 features. The first 64 features are an 8x8 image and the last two features are a (0,1) and a float. The image itself is a chessboard and consists of elements taken from the categorical set {0..12} where 0 is an empty square and 1..12 are the 6 types of pieces for each player. The (0,1) feature is a flag that indicates which player is to move next and the float is the value of the position as evaluated by Stockfish.

input = Input(shape=(40, 134))
x = LSTM(256, return_sequences=True)(input)
x = LSTM(128)(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#The output is a pair of 2 vectors of length 48, using a one-hot encoding. Each vector is the player's rating. The first vector is for white, and the seocnd is for black. The ratings are in the range 0..47, where 0 is the lowest rating and 47 is the highest rating. The ratings are integers and are distributed uniformly in the range 0..47.

#output1 = Dense(48, activation='softmax',name="WhiteElo")(x) #use if using one-hot encoding
#output2 = Dense(48, activation='softmax',name="BlackElo")(x) #use if using one-hot encoding

output1 = Dense(1,activation='relu',name="WhiteElo")(x)
output2 = Dense(1,activation='relu',name="BlackElo")(x)

model = keras.Model(inputs=input, outputs=[output1, output2])

#model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy','accuracy'])  #use if using one-hot encoding

#set metrics to mean squared error for regression

#model.compile(optimizer='adam',loss = 'mse', metrics=['accuracy','accuracy'])
#set metrics to mean squared error for regression
model.compile(optimizer='adam',loss = 'mse', metrics=['mse','mse'])

print("output shape:",model.output_shape)

model.fit(GameSequence(TRAININGDIR, batch_size=batch_size,max_queue_size=128,workers=10,shuffle=True),
           epochs=epochs,
           callbacks=[tf.keras.callbacks.ModelCheckpoint('model.keras', save_best_only=True,mode='auto',monitor='val_loss'),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')],
           validation_data=GameSequence(VALIDATIONDIR, batch_size=batch_size,max_queue_size=128,workers=10,shuffle=False),
           verbose=1)

model.save('model.keras')


