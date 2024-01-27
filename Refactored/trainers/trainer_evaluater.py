import tensorflow as tf
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
from models.simulator import create_simulator







print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




#custom training loop
@tf.function
def train_step(batch,model,evaluater,simulator):
    with tf.GradientTape() as tape_model, tf.GradientTape() as tape_evaluater:
        x,y_true = batch
        loss_model = tf.zeros((x[0].shape[0],),dtype=tf.float32)
        #all non negative values should be 1
        mask = tf.cast(tf.math.greater_equal(y_true,0),tf.float32) 
        y_true = tf.nn.relu(y_true)
        y_pred = model(x)
        #loss = tf.keras.losses.categorical_crossentropy(tf.stop_gradient(y_true),y_pred)
        loss_model = loss_model + tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),logits=y_pred) 


        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32),axis=-1)
  


        lm = tf.reduce_sum(mask*tf.keras.layers.Softmax()(y_pred),axis=-1)

        gradients = tape_model.gradient(loss_model, model.trainable_variables)
        #clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 10000)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #gradients = tape_evaluater.gradient(loss_model, evaluater.trainable_variables)
        #clip gradients
        #gradients, _ = tf.clip_by_global_norm(gradients, 10000)
        #evaluater.optimizer.apply_gradients(zip(gradients, evaluater.trainable_variables))



        return loss_model,lm,acc

from train_simu2 import myMetric
def trainer(params):
    def aux(*args, **kwargs):
        return train(*args,**kwargs, lr_start=params['lr_start'], lr = params['lr'])
    return aux

def train(gen, model,evaluater, num_step, lr_start, lr, warmup_steps):
    simulator = tf.keras.models.load_model('simulator.h5',custom_objects={'myMetric':myMetric})
    #simulator.load_weights('Refactored/simulator.h5')
    #create a log file where we will store the results. It shall be named after the current date and time
    log_file = open(f"Refactored/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w")
    summary = tf.summary.create_file_writer(f"tensorboard/logdir_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/final")
    total_steps = 0
    for epoch in range(10000):
        timer = time.time()
        total_loss = 0
        Legal_prob = 0
        accuracy = 0
        if epoch%40==0 and epoch!=0:
            #save weights
            model.save_weights(f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.h5")
        print(f"Epoch {epoch+1}:")

        for step in range(num_step):
            batch = next(gen)
            active_lr_float = (lr_start + (lr - lr_start) * min(1, (total_steps + 1) / warmup_steps))*(1/ 10**(np.floor(epoch/100)))
            
            #active_lr_float = lr_start
            model.optimizer.lr.assign(active_lr_float)
            step_var = tf.Variable(step+num_step*epoch,dtype = tf.int64)
            loss,lm,acc = train_step(batch, model,evaluater,simulator)
            loss = tf.reduce_mean(loss)
            lm = tf.reduce_mean(lm)
            total_loss = total_loss / (step + 1) * step + loss / (step + 1)
            Legal_prob = Legal_prob / (step + 1) * step + lm / (step + 1)
            accuracy = accuracy / (step + 1) * step + acc / (step + 1)


            with summary.as_default():
                tf.summary.scalar('Legal_prob',Legal_prob,step=step_var)
                tf.summary.scalar('loss', total_loss, step=step + 1000*epoch)
                tf.summary.scalar('accuracy',accuracy,step=step_var)
            #plot all accuracies on the same graph


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
        
