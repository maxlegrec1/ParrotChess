import tensorflow as tf
import numpy as np
from tqdm import tqdm
from Refactored.chessparser import *
from datetime import datetime
import multiprocessing as mp
import time
def residual(x,num_filter):
    skip= x
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    ####ADD SE BLOCK HERE
    maxpool = tf.keras.layers.GlobalAveragePooling2D()(x)
    maxpool = tf.keras.layers.Dense(32,activation='relu')(maxpool)
    maxpool = tf.keras.layers.Dense(2*num_filter,activation='relu')(maxpool)
    #split the maxpool into two parts
    W = tf.keras.layers.Lambda(lambda x: x[...,:num_filter])(maxpool)
    B = tf.keras.layers.Lambda(lambda x: x[...,num_filter:])(maxpool)
    W = tf.keras.layers.Reshape((1,1,num_filter))(W)
    B = tf.keras.layers.Reshape((1,1,num_filter))(B)
    Z = tf.keras.activations.sigmoid(W)
    x = x*Z + B
    ####END SE BLOCK

    x = tf.keras.layers.add([skip, x])
    x = tf.keras.layers.ReLU()(x)
    return x

from Bureau.projects.ParrotChess.Refactored.old.create_transformer import Encoder
h = 12  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 256  # Dimensionality of the model sub-layers' outputs
n = 10  # Number of layers in the encoder stack
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
enc_vocab_size = 12*64 + 2 # Vocabulary size for the encoder
input_seq_length = 33  # Maximum length of the input sequence
transformer = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

def create_A0(num_residuals,use_transformer = True):

    vocab = tf.keras.layers.Input(shape=(33))
    mask = tf.keras.layers.Input(shape=(33,33))
    input = tf.keras.Input(shape=(8, 8, 14))

    x = input
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(num_residuals):
        x = residual(x, 256)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    if use_transformer:
        y = transformer(vocab, mask, True)
        y = tf.keras.layers.Flatten()(y)
        x = tf.keras.layers.concatenate([x,y])

    x = tf.keras.layers.Dense(8192,activation='relu')(x)
    x = tf.keras.layers.Dense(4096,activation='relu')(x)
    x = tf.keras.layers.Dense(1792,activation='relu')(x)
    x = tf.keras.layers.Softmax()(x)
    output = x

    if use_transformer:
        return tf.keras.Model(inputs=[input,vocab,mask], outputs=output)
    else:
        return tf.keras.Model(inputs=input, outputs=output)

def only_transformer():
    vocab = tf.keras.layers.Input(shape=(33))
    mask = tf.keras.layers.Input(shape=(33,33))
    y = transformer(vocab, mask, True)
    y = tf.keras.layers.Flatten()(y)
    output = tf.keras.layers.Dense(4096,activation='relu')(y)
    output = tf.keras.layers.Dense(4096,activation='relu')(output)
    output = tf.keras.layers.Softmax()(output)
    return tf.keras.Model(inputs=[vocab,mask], outputs=output)


batch_size = 32
pgn = "../../Gigachad/pros.pgn"

#from Create_Leela import create_leela

#generator = create_leela()
#generator = only_transformer()
#generator.summary()

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
from Bureau.projects.ParrotChess.Refactored.chessparser import *
#gen = generator_uniform(generate_batch(batch_size,pgn,use_transformer=False,only_white=True),batch_size)




#generator.compile(optimizer=optimizer)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




#custom training loop
@tf.function
def train_step(batch,model):
    with tf.GradientTape() as tape:
        x,y_true = batch
        #all non negative values should be 1
        mask = tf.cast(tf.math.greater_equal(y_true,0),tf.float32) 
        y_true = tf.nn.relu(y_true)
        y_pred = model(x)
        #loss = tf.keras.losses.categorical_crossentropy(tf.stop_gradient(y_true),y_pred)
        loss =tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),logits=y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        #clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 10000)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        lm = tf.reduce_sum(mask*tf.keras.layers.Softmax()(y_pred),axis=-1)



        #tf.print(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32),axis=-1)

        return loss,lm,acc





def train(num_step, generator,gen):

    #create a log file where we will store the results. It shall be named after the current date and time
    log_file = open(f"Refactored/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w")
    total_steps = 0

    for epoch in range(10000):
        timer = time.time()
        total_loss = 0
        Legal_prob = 0
        accuracy = 0
        if epoch%40==0 and epoch!=0:
            #save weights
            generator.save_weights(f"generator_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.h5")
        print(f"Epoch {epoch+1}:")

        for step in range(num_step):
            batch = next(gen)
            active_lr_float = (lr_start + (lr - lr_start) * min(1, (total_steps + 1) / warmup_steps))*(1/ 10**(np.floor(epoch/100)))
            
            #active_lr_float = lr_start
            optimizer.lr.assign(active_lr_float)
            loss,lm,acc = train_step(batch, generator)
            loss = tf.reduce_mean(loss)
            lm = tf.reduce_mean(lm)
            total_loss = total_loss / (step + 1) * step + loss / (step + 1)
            Legal_prob = Legal_prob / (step + 1) * step + lm / (step + 1)

            accuracy = accuracy / (step + 1) * step + acc / (step + 1)
            total_steps += 1
            # Use carriage return to overwrite the current line
            print(
                f"Step: {step}, Lr: {active_lr_float:.4f}, Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Legal_prob: {Legal_prob:.4f}, time : {(time.time() - timer):.1f}"
                ,end="\r")
        # Write the results to the log file
        log_file.write(
            f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Legal_prob: {Legal_prob:.4f}\n"
        )
        log_file.flush()
        print()  # Move to the next line after completing the epoch
        

#tf.print("loading weights")
#generator.load_weights(f"generator_2023-09-03_12-56-42_120.h5")
#tf.print("weights loaded")

#tf.print("going back to previous batches")
#for i in tqdm(range(120*1000)):
    #next(gen)
#train(1000, generator)


