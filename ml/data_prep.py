from .config import *
from .shared import get_game_tensor
import numpy as np
import h5py
import zstandard as zstd
from io import TextIOWrapper
import os
import argparse

####################################################

CHUNKSIZE = 1000 #number of games to write at a time to the file

def write_to_hdf5(reader,path=""):
    """writes the games in the reader to an hdf5 file. The reader is a generator that yields game strings. The games are stored in the file according to the time-control of the game. We will write the game tensors as a dataset in the file. We will also write the ratings of the players as a dataset in the file. The file will be named according to the time-control of the games."""

    #open all the files so that we don't have to keep doing it.
    files = {}
    for file_name in set(file_dict.values()):
        files[file_name] = h5py.File(os.path.join(path,f"{file_name}.hdf5"),"a") #5*10^8 bytes = 500MB for the cache for each file           

    file_indexes = {}
    if files[file_name].get("game_tensors") is not None:
        file_indexes = {file_name:len(files[file_name]["game_tensors"]) for file_name in files}
    else:
        file_indexes = {file_name:0 for file_name in files}

    game = ""
    count = 0

    for line in reader:
        if line.startswith("[Event") and game == "": #start of a new game when the file hasn't been initialized
            game = line
        elif line.startswith("[Event") and game != "": #start of a new game when the file has been initialized, write the previous game to the file

            if count % 1000 == 0:
                print("read",count,"games")
            count += 1

            game_tensors = get_game_tensor(game)
            if game_tensors is None:
                game = line
                continue
            else:
                #print("read game",game)
                
                gt1,gt2,white_rating,black_rating,file_name = game_tensors
                #print(np.array(gt1.shape),np.array([white_rating]).shape)
                f = files[file_name]
                if f.get("game_tensors") is None:
                    f.create_dataset("game_tensors",shape=(CHUNKSIZE,NUM_MOVES,136),maxshape=(None,NUM_MOVES,136),chunks=True,compression='lzf')#,compression_opts=1)
                    f.create_dataset("ratings",shape=(CHUNKSIZE,1),chunks=True,maxshape=(None,1))#,compression='gzip',compression_opts=9)
                    f["game_tensors"][0] = gt1
                    f["game_tensors"][1] = gt2
                    f["ratings"][0] = np.array([white_rating])
                    f["ratings"][1] = np.array([black_rating])
                    file_indexes[file_name] = 2
                else: #file already exists
                    #check if we need to resize the dataset
                    if file_indexes[file_name]+1 >= f["game_tensors"].shape[0]:
                        print("enlarging chunk for file",file_name)
                    #+1 as we are writing 2 games at a time
                        f["game_tensors"].resize((f["game_tensors"].shape[0] + CHUNKSIZE,NUM_MOVES,136))
                        f["ratings"].resize((f["ratings"].shape[0] + CHUNKSIZE,1))
                    #write the new game
                    f["game_tensors"][file_indexes[file_name]] = gt1
                    f["game_tensors"][file_indexes[file_name]+1] = gt2
                    f["ratings"][file_indexes[file_name]] = np.array([white_rating])
                    f["ratings"][file_indexes[file_name]+1] = np.array([black_rating])
                    file_indexes[file_name] += 2
                game = line
        else: #continue reading the game
            game += line

    for file_name in files:
        f = files[file_name]
        #reshape the datasets to remove the extra space
        f["game_tensors"].resize((file_indexes[f],NUM_MOVES,136))
        f["ratings"].resize((file_indexes[f],1))
        f.close()

####################################################
def read_file(fn):
    #if the filename ends in .pgn we will read it as a text file. If it ends in .zst we will read it as a compressed file using streaming.
    if fn.endswith(".pgn"):
        with open(fn,"r") as f:
            write_to_hdf5(f)
    elif fn.endswith(".zst"):
        with open(fn,"rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = TextIOWrapper(reader, encoding='utf-8')
                write_to_hdf5(text_stream)


####################################################

def create_bins(f,start_index,end_index,path,**kwargs): 
    #read the hdf file up to some index and bins index values into num_bins bins based on the rating. These bins are in intervals of BIN_INTERVAL. We will store the data in the bins in separate files in the path directory.
    #N.B., f is a h5py file object. We will create a set of files under path containing the data in the bins.
    min_rating = kwargs.get("min_rating",900)
    max_rating = kwargs.get("max_rating",2500)
    BIN_INTERVAL = kwargs.get("bin_interval",50)

    #if the path doesn't exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    ratings = f["ratings"][start_index:end_index]
    
    num_bins = int((max_rating//BIN_INTERVAL)-(min_rating//BIN_INTERVAL))
    start_bin_rating = (min_rating//BIN_INTERVAL)*BIN_INTERVAL

    files = [h5py.File(f"{path}/bin_{i}.hdf5","w") for i in range(num_bins)]
    for fl in files:
        fl.create_dataset("game_tensors",shape=(0,NUM_MOVES,136),maxshape=(None,NUM_MOVES,136),compression='lzf',chunks=True)
        fl.create_dataset("ratings",shape=(0,1),maxshape=(None,1),chunks=True)

    for i in range(len(ratings)):
        bin=0
        r=f["ratings"][i][0]
        if r<=min_rating:
            bin=0
        elif r>=max_rating:
            bin=num_bins-1
        else:
            bin=int((r-start_bin_rating)//BIN_INTERVAL)
        files[bin]["game_tensors"].resize((files[bin]["game_tensors"].shape[0]+1,NUM_MOVES,136))
        files[bin]["ratings"].resize((files[bin]["ratings"].shape[0]+1,1))
        files[bin]["game_tensors"][-1] = f["game_tensors"][i]
        files[bin]["ratings"][-1] = f["ratings"][i]
        if i%100==0:
            print(f"done {i}")
    
    for fl in files:
        fl.close()

####################################################

#split_file(ORIGDATA,TESTDATA,int(num_tensors*(split[0]+split[1]),num_tensors))
def split_file(original_file_path,new_file_path,start_index,end_index):
    #splits the original file by extracting the data from start_index to end_index and writing it to a new file.
    with h5py.File(original_file_path,"r") as f:
        with h5py.File(new_file_path,"w") as nf:
            nf.create_dataset("game_tensors",data=f["game_tensors"][start_index:end_index])
            nf.create_dataset("ratings",data=f["ratings"][start_index:end_index])


####################################################
"""Calling this as a main function has the following options:
  - read_file: reads the file and appends the data to the hdf5 files as per config.py.
  - split: splits the file into training, validation and test files and creates bins from the training file.
"""  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["read_file","split"])
    parser.add_argument("file",help="The path to the input file.") #the path to the file to read
    parser.add_argument("path","Path to where outputs will be written") #the path to the directory to write the output to
    parser.add_argument("--training",type=int,default=0.8,help="Training split (between 0 an 1)") #the percentage of the data to use for training
    parser.add_argument("--validation",type=int,default=0.1,help="Validation split (between 0 and 1)") #the percentage of the data to use for validation
    parser.add_argument("--test",type=int,default=0.1,help="Testing split (between 0 and 1)") #the percentage of the data to use for testing
    parser.add_argument("--min_rating",type=int,default=900,help="Minimum rating to bin. Anything lower will go into the lowest bin") #the minimum rating to consider
    parser.add_argument("--max_rating",type=int,default=2500,help="Maximum rating to bin. Anything higher will go into the highest bin") #the maximum rating to consider
    args = parser.parse_args()

    #check that the path exists and create it if it doesn't
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    #check that the file exists
    if not os.path.exists(args.file):
        print("Error: the file",args.file,"does not exist")
        exit(1)

    if args.action == "read_file":
        read_file(args.file,args.path)
    elif args.action == "split":
        #check that args.training + args.validation + args.test = 1
        if args.training + args.validation + args.test != 1:
            print("Error: the sum of the training, validation and test percentages should be 1")
            exit(1)
        
        with h5py.File(args.file,"r") as f:
            num_tensors = f["game_tensors"].shape[0]
            split = (int(num_tensors*args.training),int(num_tensors*args.validation),int(num_tensors*args.test))
            split_file(args.file,args.path+"/training.hdf5",0,split[0])
            split_file(args.file,args.path+"/validation.hdf5",split[0],split[0]+split[1])
            split_file(args.file,args.path+"/test.hdf5",split[0]+split[1],split[0]+split[1]+split[2])
            create_bins(f,0,split[0],args.min_rating,args.max_rating,args.path)
            #delete the original training file
            os.remove(args.file)
            os.remove(args.path+"/training.hdf5")
        

    
