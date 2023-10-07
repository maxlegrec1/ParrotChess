

# from chunkparsefunc import parse_function
# from chunkparser import ChunkParser
import argparse
import os
import sys
import glob
import gzip
import random
import multiprocessing as mp




def get_latest_chunks(path, num_chunks, allow_less, sort_key_fn):
    chunks = get_all_chunks(path)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)),
                  end='',
                  flush=True)
            chunks.sort(key=sort_key_fn, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            random.shuffle(chunks)
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='', flush=True)
    chunks.sort(key=sort_key_fn, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks



def get_all_chunks(path):
    if isinstance(path, list):
        print("getting chunks for", path)
        chunks = []
        for i in path:
            chunks += get_all_chunks(i)
        return chunks
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")




def get_input_mode():
    import proto.net_pb2 as pb
    input_mode = 'classic'

    if input_mode == "classic":
        return pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    elif input_mode == "frc_castling":
        return pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
    elif input_mode == "canonical":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
    elif input_mode == "canonical_100":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
    elif input_mode == "canonical_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
    elif input_mode == "canonical_v2":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
    elif input_mode == "canonical_v2_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
    else:
        raise ValueError("Unknown input mode format: {}".format(input_mode))

"""train_chunks = get_latest_chunks("F:/Gigachad/TrainingData/Training/*/",
int(26201*0.9), False, os.path.getmtime)
"""

if __name__ == "__main__":
    """
    train_parser = ChunkParser(train_chunks,
                                get_input_mode(),
                                shuffle_size=524288,
                                sample=32,
                                batch_size=256,
                                diff_focus_min=1,
                                diff_focus_slope=0,
                                diff_focus_q_weight=6.0,
                                diff_focus_pol_scale=3.5,
                                workers=None)


    import tensorflow as tf
    train_dataset = tf.data.Dataset.from_generator(
        train_parser.parse,
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.prefetch(4)
    train_iter = iter(train_dataset)
"""
    def make_gen(generator):
        white_nums = 0
        tot = 0
        while True:
            x,y,z,q,m = next(generator)
            #transpose x to NHWC
            x = tf.transpose(x, [0, 2, 3, 1])
            #x = tf.concat([x[:,:,:,:12],tf.expand_dims(x[:,:,:,108],axis=-1)],axis=-1)
            #calculate number of white moves, x[:,:,:,108] is the white move plane, all ones when its blacks turn
            #whites = tf.reduce_sum(x[:,:,:,12],[1,2])
            #whites = tf.greater_equal(whites,0)
            #whites = tf.cast(whites, tf.float32)
            #whites = tf.reduce_sum(whites)
            #white_nums +=whites
            #tot += 256
            #tf.print(white_nums/tot)
            yield(x,y)
    #gen = make_gen(train_iter)
    from Bureau.projects.ParrotChess.Refactored.chessparser import *
    batch_size = 256
    pgn = "/home/antoine/Bureau/projects/ParrotChess/Refactored/human.pgn"
    gen = generator_uniform(generate_batch(batch_size,pgn,use_transformer=False,only_white=True),batch_size)
    from Resnet import *
    from Create_Leela import create_leela
    generator = create_leela()
    #generator = only_transformer()
    generator.summary()

    lr = 2e-2
    warmup_steps = 2000
    lr_start = 1e-5
    active_lr = tf.Variable(lr_start, dtype=tf.float32,trainable=False)
    #optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.9, beta_2=0.98,
    #                                     epsilon=1e-9)
    optimizer = tf.keras.optimizers.SGD(
                        learning_rate=active_lr,
                        momentum=0.9,
                        nesterov=True)
    generator.compile(optimizer=optimizer)

    x,y = next(gen)
    print(x.shape)
    train(1000, generator,gen)