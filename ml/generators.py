from tensorflow.keras.utils import Sequence
import h5py
import numpy as np
import os
import random
from config import NUM_MOVES

class InMemoryOverSamplngGenerator(Sequence):
    def __init__(self,path,batch_size,**kwargs):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = kwargs.get("shuffle",True)
        self.num_items = kwargs.get("num_items",None)

        self.files = [h5py.File(f"{path}/bin_{i}.hdf5","r") for i in range(len(os.listdir(path)))]
                      
        self.bins = [[] for i in range(len(self.files))]
        self.num_bins = len(self.bins)

        #read the data into memory
        for i in range(len(self.files)):
            self.bins[i] = (self.files[i]["game_tensors"][:],self.files[i]["ratings"][:])
        
        self.current_bin = 0
        self.bin_indexes = [0 for i in range(self.num_bins)]

        for f in self.files:
            f.close()

    def __len__(self):
        if self.num_items is None:
            return sum([len(b[0]) for b in self.bins])//self.batch_size
        else:
            return self.num_items//self.batch_size

    def __getitem__(self,index):
        x_batch = []
        y_batch = []

        num_items = self.batch_size
        while num_items > 0:
            if self.bin_indexes[self.current_bin] == len(self.bins[self.current_bin][0]):
                self.bin_indexes[self.current_bin] = 0
                self.current_bin += 1
                self.current_bin %= self.num_bins

            x_batch.append(self.bins[self.current_bin][0][self.bin_indexes[self.current_bin]])
            y_batch.append(self.bins[self.current_bin][1][self.bin_indexes[self.current_bin]])

            self.bin_indexes[self.current_bin] += 1
            num_items -= 1

        return np.array(x_batch),np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            for i in range(self.num_bins):
                state = random.random.get_state()
                seed = random.randint(0,10000)
                random.seed(seed)
                random.shuffle(self.bins[i][0])
                random.seed(seed)
                random.shuffle(self.bins[i][1])
                random.random.set_state(state)
    


class TrainingGenerator(Sequence):
    """This generator takes in a path containing a set of hdf5 files, each of which is a bin. The generator will yield data by taking batch_size elements from each bin in the path. We will store cache_size elements from each file in memory, loading them in as needed. The generator will load the data from the files in the path in order, and will loop back to the start when it reaches the end of the files. The generator will also shuffle the order of the files if shuffle is set to True."""
    def __init__(self,path,batch_size,**kwargs):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = kwargs.get("shuffle",True)
        self.cache_size = kwargs.get("cache_size",128)
        self.num_items = kwargs.get("num_items",None)
        

        self.files = [h5py.File(f"{path}/bin_{i}.hdf5","a") for i in range(len(os.listdir(path)))]
        self.num_files = len(self.files)
        self.file_indexes = [0 for i in range(self.num_files)]
        self.game_cache = [np.zeros((self.cache_size,NUM_MOVES,136),dtype=np.int16) for i in range(self.num_files)]
        self.rating_cache = [np.zeros((self.cache_size,1),dtype=np.int16) for i in range(self.num_files)]

        self.cache_index = [0 for i in range(self.num_files)]

        self.current_file = 0

    def __len__(self):
        """returns the number of batches in the generator. This is the sum of the number of elements in each file divided by the batch size."""
        if self.num_items is None:
            return sum([len(f["ratings"]) for f in self.files])//self.batch_size
        else:
            return self.num_items//self.batch_size
    
    def __getitem__(self,index):
        """returns the next batch. The batch is a tuple containing the game tensors and the ratings. We ignore the index as we will just iterate through the files in order."""
        x_batch = []
        y_batch = []

        num_items = self.batch_size

        while num_items>0:
            #check if we need to load more data into the cache
            if self.cache_index[self.current_file] == 0:
                self.__load_data(self.current_file)

            #get the next element from the cache
            x_batch.append(self.game_cache[self.current_file][self.cache_index[self.current_file]])
            y_batch.append(self.rating_cache[self.current_file][self.cache_index[self.current_file]])

            self.cache_index[self.current_file] += 1
            self.cache_index[self.current_file] %= self.cache_size
            self.current_file += 1
            self.current_file %= self.num_files
            num_items -= 1

        return np.array(x_batch),np.array(y_batch)
    
    def __load_data(self,file_index):
        """loads the next cache_size elements from the file at file_index into the cache. If the end of the file is reached, the cache loops around. """
        f = self.files[file_index]
        start_index = self.file_indexes[file_index]
        end_index = start_index + self.cache_size

        #print(f"reading from file {f.filename} from {start_index} to {end_index}, cache size is {self.cache_size}")

        if end_index > len(f["ratings"]):
            num_read = len(f["ratings"]) - start_index
            self.game_cache[file_index][:num_read] = f["game_tensors"][start_index:]
            self.rating_cache[file_index][:num_read] = f["ratings"][start_index:]

            #print(f"I have read {num_read} elements from file, about to read {self.cache_size-num_read} elements from the start of the file")

            self.game_cache[file_index][num_read:] = f["game_tensors"][:self.cache_size-num_read]
            self.rating_cache[file_index][num_read:] = f["ratings"][:self.cache_size-num_read]

            self.file_indexes[file_index] = end_index % len(f["ratings"])
        else:
            self.game_cache[file_index] = f["game_tensors"][start_index:end_index]
            self.rating_cache[file_index] = f["ratings"][start_index:end_index]
            self.file_indexes[file_index] = end_index
    

    def on_epoch_end(self):
        """shuffles the order of the files if shuffle is set to True. Also cleares the caches and resets the file indexes."""
        self.file_indexes = [0 for i in range(self.num_files)]
        self.game_cache = [np.zeros((self.cache_size,NUM_MOVES,136),dtype=np.int16) for i in range(self.num_files)]
        self.rating_cache = [np.zeros((self.cache_size,1),dtype=np.int16) for i in range(self.num_files)]
        self.cache_index = [0 for i in range(self.num_files)]
        self.current_file = 0

        if self.shuffle:
            print("shuffling files")    
            #save current random number generator state
            prng_state = random.random.get_state()
            
            for f in self.files:
                print("shuffling file ",f.filename)
                seed = random.randint(0,10000)
                random.seed(seed)
                random.shuffle(f["ratings"])
                random.seed(seed)
                random.shuffle(f["game_tensors"])
            
            #restore the random number generator state
            random.random.set_state(prng_state)
            print("done shuffling files")


    def __del__(self):
        for f in self.files:
            f.close()

class InMemoryGenerator(Sequence):
    def __init__(self,file,batch_size,shuffle=False):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle=shuffle

        with h5py.File(file,"r") as f:
            self.game_tensors = f["game_tensors"][:]
            self.ratings = f["ratings"][:]

    def __len__(self):
        return len(self.ratings)//self.batch_size

    def __getitem__(self, index):
    
        x_batch=[]
        y_batch=[]

        for i in range(self.batch_size):
            x_batch.append(self.game_tensors[(index*self.batch_size+i)%len(self.game_tensors)])
            y_batch.append(self.ratings[(index*self.batch_size+i)%len(self.ratings)])
        
        return np.array(x_batch),np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            state = random.random.get_state()
            seed = random.randint(0,10000)
            random.seed(seed)
            random.shuffle(self.game_tensors)
            random.seed(seed)
            random.shuffle(self.ratings)
            random.random.set_state(state)

        
class HDF5FileGenerator(Sequence):
    def __init__(self, file, batch_size, shuffle=False):
        super().__init__()
        self.f = h5py.File(file,"a",rdcc_nbytes=5*10**8) #500MB cache
        self.batch_size = batch_size
        self.shuffle=shuffle
        
    def __len__(self):
        return (len(self.f["ratings"]))//self.batch_size
        

    def __getitem__(self, index):

        x_batch=[]
        y_batch=[]

        for i in range(self.batch_size):
            x_batch.append(self.f["game_tensors"][(index*self.batch_size+i)%len(self.f["game_tensors"])])
            y_batch.append(self.f["ratings"][(index*self.batch_size+i)%len(self.f["ratings"])])
        
        return np.array(x_batch),np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            state = random.random.get_state()
            seed = random.randint(0,10000)
            random.seed(seed)
            random.shuffle(self.f["game_tensors"])
            random.seed(seed)
            random.shuffle(self.f["ratings"])
            random.random.set_state(state)

    def __del__(self):
        self.f.close()