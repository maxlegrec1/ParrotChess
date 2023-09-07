import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime

def residual(x,num_filter):
    skip= x
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([skip, x])
    x = tf.keras.layers.ReLU()(x)
    return x
def denseNet(list_of_layers):
    def call(x):
        for layer in list_of_layers:
            
            x = tf.keras.layers.Dense(**layer)(x)
        return x
    return call
def ResNet(num_residuals):
    def call(x):
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        for _ in range(num_residuals):
            x = residual(x, 256)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu')(x)
        return x
    return call
    
from create_transformer import Encoder
h = 16  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 256  # Dimensionality of the model sub-layers' outputs
n = 10  # Number of layers in the encoder stack
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
enc_vocab_size = 12*64 # Vocabulary size for the encoder
input_seq_length = 32  # Maximum length of the input sequence
transformer = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

def create_A0(num_residuals):
    vocab = tf.keras.layers.Input(shape=(32))
    mask = tf.keras.layers.Input(shape=(32,32))
    y = transformer(vocab, mask, True)
    y = tf.keras.layers.Flatten()(y)
    input = tf.keras.Input(shape=(8, 8, 14))
    x = input
    
    ResNet0 = ResNet(num_residuals)
    DenseNet0 = denseNet([{'units':8192, 'activation': 'relu'},{'units':8192, 'activation': 'relu'},{'units':4096, 'activation': 'relu'}])
    
    x = ResNet0(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.concatenate([x,y])
    
    x = DenseNet0(x)
    x = tf.keras.layers.Softmax()(x)
    output = x
    return tf.keras.Model(inputs=[input,vocab,mask], outputs=output)


generator = create_A0(40)
generator.summary()


from refactored2 import *
gen = generate_batch()





optimizer = tf.keras.optimizers.Adam(1e-6, beta_1=0.9, beta_2=0.99,
                                     epsilon=1e-9)



generator.compile(optimizer=optimizer)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#Loading old model weights
# generator.load_weights('generator_2023-09-06_13-53-48_80.h5')

#custom training loop
@tf.function
def train_step(batch,model,metric5,metric10,masked5,masked10,metric1,masked1):
    with tf.GradientTape() as tape:
        x,y_true,mask = batch
        y_pred = model(x)
        false_moves = (1-mask)*y_pred
        loss1 = tf.keras.losses.CategoricalCrossentropy()(y_true,y_pred)
        loss2 = tf.keras.losses.MeanAbsoluteError()(false_moves,tf.zeros_like(false_moves))
        loss = loss1
        legal_prob = tf.reduce_sum(mask*y_pred,axis=-1)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        masked_pred = tf.keras.layers.multiply([y_pred,mask])
        metric5.update_state(y_true,y_pred)
        metric10.update_state(y_true,y_pred)

        masked5.update_state(y_true,masked_pred)
        masked10.update_state(y_true,masked_pred)

        metric1.update_state(y_true,y_pred)
        masked1.update_state(y_true,masked_pred)

        return loss1,loss2,legal_prob





def train(num_step, generator):
    masked1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name = "masked_Acc")
    masked5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name = "masked5")
    masked10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name = "masked10")
    metric1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name = "Accuracy")
    metric5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name = "top5")
    metric10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10,name = "top10")

    #create a log file where we will store the results. It shall be named after the current date and time
    log_file = open(f"Refactored/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w")

    for epoch in range(10000):
        total_loss1 = 0
        total_loss2 = 0
        total_legal_prob = 0
        metric5.reset_state()
        metric10.reset_state()
        masked5.reset_state()
        masked10.reset_state()
        masked1.reset_state()
        metric1.reset_state()
        if epoch%40==0:
            #save weights
            generator.save_weights(f"generator_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.h5")
        print(f"Epoch {epoch+1}:")

        for step in range(num_step):
            batch = next(gen)
            loss1, loss2, legal_prob = train_step(batch, generator, metric5, metric10,masked5,masked10,metric1,masked1)
            loss1, loss2, legal_prob = tf.reduce_mean(loss1), tf.reduce_mean(loss2), tf.reduce_mean(legal_prob)
            total_loss1 = total_loss1 / (step + 1) * step + loss1 / (step + 1)
            total_loss2 = total_loss2 / (step + 1) * step + loss2 / (step + 1)
            total_legal_prob = total_legal_prob / (step + 1) * step + legal_prob / (step + 1)
            
            accuracy = metric1.result().numpy()
            metric5_result = metric5.result().numpy()
            metric10_result = metric10.result().numpy()
            masked5_result = masked5.result().numpy()
            masked10_result = masked10.result().numpy()
            masked1_result = masked1.result().numpy()
            # Use carriage return to overwrite the current line
            print(
                f"Step: {step}, Loss1: {total_loss1:.4f}, Loss2: {total_loss2:.6f},  Acc: {accuracy:.4f}, Masked-Acc: {masked1_result:.4f}, Legal_prob: {total_legal_prob:.4f},\n "
                f"Top-5: {metric5_result:.4f}, Masked-5: {masked5_result:.4f}, "
                f"Top-10: {metric10_result:.4f}, Masked-10: {masked10_result:.4f}",
                end="\r"  # Return to the beginning of the line
            )
            print()
        # Write the results to the log file
        log_file.write(
            f"Epoch: {epoch + 1}, Loss1: {total_loss1:.4f}, Loss2: {total_loss2:.6f}, Acc: {accuracy:.4f}, Masked-Acc: {masked1_result:.4f}, Legal_prob: {total_legal_prob:.4f}, "
            f"Top-5: {metric5_result:.4f}, Masked-5: {masked5_result:.4f}, "
            f"Top-10: {metric10_result:.4f}, Masked-10: {masked10_result:.4f}\n"
        )
        log_file.flush()
        print()  # Move to the next line after completing the epoch




train(1000, generator)




