import keras

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
#from keras.layers import CategoryEncoding
import keras_tuner as kt


import os,io

import tensorflow as tf

import chess
import chess.pgn


import numpy as np

NUM_MOVES = 40 #number of moves to store in tensor

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

def get_game_tensors(game_pgn,num_tensors):

    gt = np.zeros((num_tensors,137))

    moves = []
    evals = []
    clock = []

    for m in game_pgn.mainline():
        moves.append(m.san())
        clock.append(m.clock())
        if m.eval() is None: #should only happen on a mate
            evals.append(None)
        else:
            evals.append(m.eval().white() if m.turn() == chess.WHITE else m.eval().black())

    #let our t vector be a 1D array of 130 elements. The first 128 element represent the board before the move is made and after the move is made. The 129th element is 1 if it is white moved and -1 if it is black making the move. The 130th element is the move number.

    board = chess.Board()

    for m in range(0,min(num_tensors,len(moves)-1)):
        t = np.zeros(137)

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
        t[134] = m #record the move number
        t[135] = clock[m]
        t[136] = clock[m+1]

        gt[m] = t

    return gt

def extract_game(game_file,start_pos):
    #read from game_file line by line. Extract the part between [Event ...] and the next [Event ...]. Return the extracted string and the position in game_file of the start of the next game.

    game = ""
    pos = start_pos
    #seek to the start_pos
    game_file.seek(pos)
    found_start = False
    found_end = False
    
    while not found_start:
        line = game_file.readline()
        if line == "":
            #print("NULL FOUND")
            return None,None
        if line.startswith("[Event "):
            found_start = True
            game += line
    
    while not found_end:
        pos = game_file.tell()
        line = game_file.readline()
        if line.startswith("[Event ") or line == "":    
            found_end = True
        else:
            game += line

    return game,pos

def check_game(string):
    #checks if there is a string 'TimeControl "600+0"' in the game string, whether the string contains the string 'eval' and whether the string 'WhiteRatingDiff "X"' and 'BlackRatingDiff "Y"' are present and the absolute values of X and Y are less than 40.

    #print("GAME STRING:",string,"END GAME STRING")
    if string == "":
        return None

    if 'TimeControl "300+0"' in string and 'eval' in string and 'WhiteRatingDiff' in string and 'BlackRatingDiff' in string:
        white_diff = int(string.split('WhiteRatingDiff "')[1].split('"')[0])
        black_diff = int(string.split('BlackRatingDiff "')[1].split('"')[0])
        if abs(white_diff) < 40 and abs(black_diff) < 40:
            return True
        
    return False

def make_data(game_file,path,target_file):
    count = 0
    found_count = 0
    X,y1,y2 = [],[],[]

    with open(game_file) as f:

        pos = 0
        while True:
            game,pos = extract_game(f,pos)
            if game is None:
                break
            if check_game(game):
                #print("Found game:",game)
                game = chess.pgn.read_game(io.StringIO(game))
                gt = get_game_tensors(game,NUM_MOVES)
                y1t = get_ratings(game)[0]
                y2t = get_ratings(game)[1]

                X.append(gt)
                y1.append(y1t)
                y2.append(y2t)
                found_count += 1
                if found_count % 1000 == 0:
                    print("Found " + str(found_count) + " games")
                    np.savez_compressed(os.path.join(path,target_file + "_X.npz"),np.array(X))
                    np.savez_compressed(os.path.join(path,target_file + "_y1.npz"),np.array(y1))
                    np.savez_compressed(os.path.join(path,target_file + "_y2.npz"),np.array(y2))


            count += 1
            if count % 100000 == 0:
                print("Read " + str(count) + " games")

    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)

    #save the data to the target file
    np.savez_compressed(os.path.join(path,target_file + "_X.npz"),X)
    np.savez_compressed(os.path.join(path,target_file + "_y1.npz"),y1)
    np.savez_compressed(os.path.join(path,target_file + "_y2.npz"),y2)

def load_data(path,target_file):
    X = np.load(os.path.join(path,target_file + "_X.npz"))["arr_0"]
    y1 = np.load(os.path.join(path,target_file + "_y1.npz"))["arr_0"]
    y2 = np.load(os.path.join(path,target_file + "_y2.npz"))["arr_0"]

    return X,y1,y2

def simplify_data_no_eval(X):
    #transforms the 134 element tensor into a 131 element tensor by removing elements 129-132 and replacing them with a single element that is 1 if white is moving and -1 if black is moving and a single element that is the move number
    Xs = np.zeros((X.shape[0],X.shape[1],132))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xs[i][j][0:64] = X[i][j][0:64]
            Xs[i][j][64:128] = X[i][j][64:128]
            Xs[i][j][128] = X[i][j][133]
            #element Xs[i][j][129] is the move number, i.e., j
            Xs[i][j][129] = j
            Xs[i][j][130] = X[i][j][135] #clock
            Xs[i][j][131] = X[i][j][136] #clock

    return Xs

def simplify_data_eval_only(X):
    #takes only elements 129-133 and adds move number
    Xs = np.zeros((X.shape[0],X.shape[1],8))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xs[i][j][0] = X[i][j][129]
            Xs[i][j][1] = X[i][j][130]
            Xs[i][j][2] = X[i][j][131]
            Xs[i][j][3] = X[i][j][132]
            Xs[i][j][4] = X[i][j][133]
            Xs[i][j][5] = j
            Xs[i][j][6] = X[i][j][135] #clock
            Xs[i][j][7] = X[i][j][136] #clock

    return Xs

#if the data doesn't exist, generate it
if not os.path.exists("data/all_data/data_X.npz"):
    make_data("data/all_data/lichess.pgn","data/all_data/","data")

X,y1,y2 = load_data("data/all_data","data")

#X = simplify_data_eval_only(X)
#X = simplify_data_no_eval(X)

def model_builder(hp):

    #inputs = Input(shape=(NUM_MOVES, 132)) #if no eval is used
    inputs = Input(shape=(NUM_MOVES, 137)) #full tensor
    #inputs = Input(shape=(NUM_MOVES,8)) #if only the eval is used
    
    x = inputs

    #prepare hyperparameter tuning

    num_LSTM_layers = hp.Int('num_LSTM_layers,0,2')
    num_LSTM_units=[]
    for i in range(num_LSTM_layers+1):
        num_LSTM_units.append(hp.Int('lstm'+str(i)+'_units',
                                     min_value = 32,
                                     max_value = 256,
                                     step=16))
        
                                     
    num_dense_layers = hp.Int('num_dense_layers',1,3)
    num_dense_units = []
    dense_activation = []

    for i in range(num_dense_layers):
        num_dense_units.append(hp.Int('dense'+str(i)+'_units',
                                     min_value = 32,
                                     max_value = 256,
                                     step=16))
        dense_activation.append(hp.Choice("dense+str(i)+_activation",["relu", "selu","leaky_relu","tanh"]))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-2])

    #make the NN

    for i in range(num_LSTM_layers):
        x = LSTM(num_dense_units[i],return_sequences = True)(x)

    #add a final LSTM layer that doesn't return sequences
    x = LSTM(num_LSTM_units[-1])(x)
    
    for i in range(num_dense_layers):
        x = Dense(num_dense_units[i],activation = dense_activation[i])(x)


    output1 = Dense(1,activation='relu',name="WhiteElo")(x)
    output2 = Dense(1,activation='relu',name="BlackElo")(x)

    model = keras.Model(inputs=inputs,outputs=[output1,output2])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss={'WhiteElo':'mae','BlackElo':'mae'},
                    metrics={'WhiteElo':'mae','BlackElo':'mae'})

    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=100,
                     factor=3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
save = tf.keras.callbacks.ModelCheckpoint('model.keras', save_best_only=True,mode='auto',monitor='val_loss')

tuner.search(X,(y1,y2),epochs=100,validation_split=0.2,callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.values)

model = tuner.hypermodel.build(best_hps)

history = model.fit(X,(y1,y2),epochs=100,validation_split=0.2,callbacks=[stop_early,save])
model.save('model3.keras')