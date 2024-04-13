import os
import sys
import random
import glob
import itertools
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GPU_ID = 0
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[GPU_ID], 'GPU')
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[GPU_ID], True)

sys.path.append(os.getcwd())
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training")
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training/tf")
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training/tf/proto")

from chunkparser import ChunkParser
from chunkparsefunc import parse_function
import proto.net_pb2 as pb


def fast_get_chunks(d):
    d = d.replace("*/", "")
    chunknames = []
    fo_chunknames = []
    subdirs = os.listdir(d)
    chunkfiles_name = "chunknames.pkl"
    if False and chunkfiles_name in subdirs: # TODO: remove False
        print(f"Using cached {d + chunkfiles_name}" )
        with open(d + chunkfiles_name, 'rb') as f:
            chunknames = pickle.load(f)
    else:

        i = 0
        for subdir in subdirs:
            if subdir.endswith(".gz"):
                fo_chunknames.append(d + subdir)
            else:
                prefix = d + subdir + "/"
                if os.path.isdir(prefix):
                    chunknames.append([prefix + s for s in os.listdir(prefix) if s.endswith(".gz")])

            i += 1
        chunknames.append(fo_chunknames)
            
        chunknames = list(itertools.chain.from_iterable(chunknames))

        with open(d + chunkfiles_name, 'wb') as f:
            print("Shuffling the chunks", flush=True)
            random.shuffle(chunknames)
            print(f"Caching {d + chunkfiles_name}" )
            pickle.dump(chunknames, f)

    return chunknames



def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_all_chunks(path, fast=False):

    if isinstance(path, list):
        print("getting chunks for", path)
        chunks = []
        for i in path:
            chunks += get_all_chunks(i, fast=fast)
        return chunks
    if fast:
        chunks = fast_get_chunks(path)
    else:
        chunks = []
        for d in glob.glob(path):
            chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks


def get_latest_chunks(path, num_chunks, allow_less, sort_key_fn, fast=False):
    chunks = get_all_chunks(path, fast=fast)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)),
                  end="",
                  flush=True)
            if False:
                print("sorting disabled")
            else:
                chunks.sort(key=sort_key_fn, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            print("shuffling chunks...", end="", flush=True)
            if True:
                print("shuffling disabled", flush=True)
            else:
                random.shuffle(chunks)
            print("[done] less")
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end="", flush=True)
    chunks.sort(key=sort_key_fn, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks



class data_gen():
    def __init__(self,*args,**kwargs):
        sort_key_fn = lambda x: x

        train_chunks = get_latest_chunks("/home/maxime/Downloads/data/*/",
                                                500_000_000, True, sort_key_fn, fast=True)

        train_parser = ChunkParser(train_chunks,
                                    pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE,
                                    shuffle_size=15_000,
                                    sample=32,
                                    batch_size=1024,
                                    diff_focus_min=1,
                                    diff_focus_slope=0,
                                    diff_focus_q_weight=6,
                                    diff_focus_pol_scale=3.5,
                               workers=4)


        train_dataset = tf.data.Dataset.from_generator(
                train_parser.parse,
                output_types=7 * (tf.string,))

        train_dataset = train_dataset.prefetch(4)
        train_dataset = train_dataset.map(parse_function)
        train_iter = iter(train_dataset)




        self.ds = train_iter
    
    def get_batch(self):
        return next(self.ds)
    
if __name__ == "__main__":
    ds =data_gen()
    arr = []
    for i in range(100):
        X,Y = ds.get_batch()
        num_pieces = tf.reduce_sum(X[:,:12,:,:],axis = [1,2,3])
        arr.append(num_pieces)
    arr = tf.concat(arr,axis = 0)
    arr = arr.numpy()
    bins = np.arange(1,33)
    plt.hist(arr,bins = bins)
    plt.show()