import tensorflow as tf

import keras
from keras.layers import Input, Dense, LSTM, TimeDistributed
import keras_tuner as kt
from .generators import TrainingGenerator, HDF5FileGenerator
from .config import NUM_MOVES
from .shared import get_game_tensor
import numpy as np
import os

import argparse


def make_generators(training_path, validation_file, test_file, **kwargs):
    batch_size = kwargs.get('batch_size',64)
    shuffle = kwargs.get('shuffle',True)
    cache_size = kwargs.get('cache_size',128)
    training_gen =TrainingGenerator(training_path,batch_size,shuffle=shuffle,cache_size=cache_size)
    validation_gen = HDF5FileGenerator(validation_file,batch_size,False)
    test_gen = HDF5FileGenerator(test_file,batch_size,False)

    return training_gen, validation_gen, test_gen

def make_model(**kwargs):
    bins = kwargs.get("bins",False)
    min_rating = kwargs.get("min_rating",900)
    max_rating = kwargs.get("max_rating",2500)
    bin_size = kwargs.get("bin_size",50)

    num_bins=(max_rating-min_rating)//bin_size


    inputs = Input(shape=(NUM_MOVES, 136)) #full tensor
    x = TimeDistributed(Dense(104,activation = 'leaky_relu'))(inputs)
    x = LSTM(36,return_sequences = True)(x)
    x = LSTM(40)(x)
    x = Dense(96,activation='leaky_relu')(x)
    x = Dense(104,activation='leaky_relu')(x)
    x = Dense(120,activation='relu')(x)
    output = None
    if not bins:
        output = Dense(1,activation='relu',name="Elo")(x)
    else:
        output = Dense(num_bins,activation='softmax',name="Elo")(x)
    model = keras.Model(inputs=inputs,outputs=[output])
    if not bins:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss={'Elo':'mae'},
                    metrics={'Elo':'mae'})
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss={'Elo':'categorical_crossentropy'},
                    metrics={'Elo':'categorical_accuracy'})
    return model

def train_model(model, train_gen, val_gen, test_gen,**kwargs):
    epochs = kwargs.get('epochs',100)
    model_filename = kwargs.get('model_filename','model.keras')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    save = tf.keras.callbacks.ModelCheckpoint(model_filename, 
                                              save_best_only=True,
                                              mode='auto',
                                              monitor='val_loss')

    model.fit(train_gen,validation_data=val_gen,epochs=100,callbacks=[stop_early,save])

    model.evaluate(test_gen)

    #N.B., the model saved with the save callback is the best model according to the validation loss. We can load this model and evaluate it on the test data.

    model = keras.models.load_model(model_filename)
    print(model.evaluate(test_gen))
    return model

def model_builder(hp):
    #it looks like the best model has 2 LSTM layers, 1 dense layer and 1 TD layer. We will use this as a starting point for the hyperparameter search.

    inputs = Input(shape=(NUM_MOVES, 136)) #full tensor
    
    x = inputs

    #prepare hyperparameter tuning

    num_LSTM_layers = hp.Int('num_LSTM_layers',2,3)
    num_LSTM_units=[]
    for i in range(num_LSTM_layers):
        num_LSTM_units.append(hp.Int('lstm'+str(i+1)+'_units',
                                     min_value = 36,
                                     max_value = 44,
                                     step=4))
        
                                     
    num_dense_layers = hp.Int('num_dense_layers',2,3)
    num_dense_units = []
    dense_activation = []

    for i in range(num_dense_layers):
        num_dense_units.append(hp.Int('dense'+str(i+1)+'_units',
                                     min_value = 96,
                                     max_value = 120,
                                     step=8))
        dense_activation.append(hp.Choice("dense"+str(i+1)+"_activation",["relu", "leaky_relu"]))
    
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-2])
    hp_learning_rate = 0.001

    #make the NN
    x = TimeDistributed(Dense(hp.Int('td_dense_units',min_value=88,max_value=120,step=8),activation=hp.Choice("td_dense_activation",["relu","leaky_relu"])))(x)

    for i in range(num_LSTM_layers):
        x = LSTM(num_LSTM_units[i],return_sequences=True if i<num_LSTM_layers-1 else False)(x)

    
    for i in range(num_dense_layers):
        x = Dense(num_dense_units[i],activation = dense_activation[i])(x)


    output = Dense(1,activation='relu',name="Elo")(x)
    

    #Alternative: set outputs to be hot encoded between 48 values
    #output1 = Dense(48,activation='softmax',name="WhiteElo")(x)
    #output2 = Dense(48,activation='softmax',name="BlackElo")(x)

    model = keras.Model(inputs=inputs,outputs=[output])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss={'Elo':'mae'},
                    metrics={'Elo':'mae'})

    return model

def tune_model(model_builder, train_gen, val_gen,**kwargs):
    filename = kwargs.get('filename','modelCP.keras')


    tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=100,
                     factor=5)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    save = tf.keras.callbacks.ModelCheckpoint(filename, save_best_only=True,mode='auto',monitor='val_loss')

    tuner.search(train_gen,validation_data=val_gen,epochs=100,callbacks=[stop_early,save])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(best_hps.values)
    model = tuner.hypermodel.build(best_hps)
    #model = keras.models.load_model(filename)

    model.fit(train_gen,validation_data=val_gen,epochs=100,callbacks=[stop_early,save])

def evaluate_model(model_file, test_gen):
    model = keras.models.load_model(model_file)
    print(model.summary())
    print(model.evaluate(test_gen))

def predict(**kwargs):
    """Takes either a file or a pgn string as input and predicts the elo of the players"""
    model = None
    model_file = kwargs.get('model_file',None)
    if model_file is not None:
        model = keras.models.load_model(model_file)
    else:
        model = kwargs.get('model',None)
    pgn_file = kwargs.get('pgn_file',None)
    pgn_string = kwargs.get('pgn_string',None)
    if pgn_file is None and pgn_string is None:
        print("Please provide a pgn file or a pgn string")
        return
    if pgn_file is not None:
        pgn_string = open(pgn_file).read()
    
    game_tensors = get_game_tensor(pgn_string,do_checks=False)
    #predict the elo
    elo = model.predict(np.array([game_tensors[0],game_tensors[1]]),batch_size=2)
    #print("elo shape:",elo.shape)
    #print("White Elo:",elo[0])
    #print("Black Elo:",elo[1])
    return elo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["train","tune","evaluate","predict"])
    parser.add_argument("training_path",help="The path to the directory with the training files") #the path to the directory containing the training data
    parser.add_argument("validation_file",help="The path to the validation file") #the path to the validation file
    parser.add_argument("test_file",help = "The path to the test file") #the path to the test file
    parser.add_argument("model_file",help="The path to the model file") #the path to the model file
    parser.add_argument("--batch_size",type=int,default=64) #the batch size
    parser.add_argument("--shuffle",type=bool,default=False) #whether to shuffle the data
    parser.add_argument("--cache_size",type=int,default=128) #the cache size used for training data
    parser.add_argument("--bin",type=bool,default=False) #whether the training data is in binned. If it is, the model will output a softmax layer
    parser.add_argument("--min_rating",type=int,default=900) #the minimum rating for the bins   
    parser.add_argument("--max_rating",type=int,default=2500) #the maximum rating for the bins
    parser.add_argument("--bin_size",type=int,default=50) #the size of the bins
    args = parser.parse_args()

    #check that the files exist
    if not os.path.exists(args.training_path):
        print(f"Training path {args.training_path} does not exist")
        exit(1)
    if not os.path.exists(args.validation_file):
        print(f"Validation file {args.validation_file} does not exist")
        exit(1)
    if not os.path.exists(args.test_file):
        print(f"Test file {args.test_file} does not exist")
        exit(1)

    
    training_gen, validation_gen, test_gen = make_generators(args.training_path,args.validation_file,args.test_file,batch_size=args.batch_size,shuffle=args.shuffle,cache_size=args.cache_size,bin=args.bin,min_rating=args.min_rating,max_rating=args.max_rating,bin_size=args.bin_size)

    if args.action == "train":
        model = make_model(bin=args.bin,min_rating=args.min_rating,max_rating=args.max_rating,bin_size=args.bin_size)
        train_model(model,training_gen,validation_gen,test_gen,model_filename=args.model_file)
    elif args.action == "tune":
        tune_model(model_builder,training_gen,validation_gen,filename=args.model_file)
    elif args.action == "evaluate":
        evaluate_model(args.model_file,test_gen)
    elif args.action == "predict":
        predict(model_file=args.model_file,pgn_file=args.pgn_file)