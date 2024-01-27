import tensorflow as tf
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
import csv





print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




#custom training loop
@tf.function
def train_step(batch,model):
    with tf.GradientTape() as tape:
        x,y_true = batch
        #all non negative values should be 1
        mask = tf.cast(tf.math.greater_equal(y_true,0),tf.float32) 
        y_true = tf.nn.relu(y_true)
        y_pred = model(x)*mask
        #loss = tf.keras.losses.categorical_crossentropy(tf.stop_gradient(y_true),y_pred)
        loss =tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),logits=y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        #clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 10000)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #tf.print(mask.dtype,y_pred.dtype)
        lm = tf.reduce_sum(tf.keras.layers.Softmax()(y_pred),axis=-1)



        #tf.print(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32),axis=-1)

        return loss,lm,acc


def trainer(params):
    def aux(*args, **kwargs):
        return train(*args,**kwargs, lr_start=params['lr_start'], lr = params['lr'])
    return aux




def train(gen, model, num_step, lr_start ,lr, warmup_steps):

    #create a log file where we will store the results. It shall be named after the current date and time
    log_file = open(f"Refactored/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", "w")
    summary = tf.summary.create_file_writer(f"tensorboard/logdir_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    names_to_save = ['epoch', 'total_loss', 'accuracy', 'Legal_prob']
    writer = csv.DictWriter(log_file, fieldnames=names_to_save)
    writer.writeheader()
    total_steps = 0
    for epoch in range(0,10000):
        timer = time.time()
        total_loss = 0
        Legal_prob = 0
        accuracy = 0
        if epoch%40==0 and epoch!=0:
            #save weights
            model.save_weights(f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.h5")
        print(f"Epoch {epoch+1}:")

        for step in range(num_step):
            batch = gen.get_batch()
            active_lr_float = (lr_start + (lr - lr_start) * min(1, (total_steps + 1) / warmup_steps))
            model.optimizer.lr.assign(active_lr_float)
            loss,lm,acc = train_step(batch, model)
            loss = tf.reduce_mean(loss)
            lm = tf.reduce_mean(lm)
            total_loss = total_loss / (step + 1) * step + loss / (step + 1)
            Legal_prob = Legal_prob / (step + 1) * step + lm / (step + 1)

            accuracy = accuracy / (step + 1) * step + acc / (step + 1)
            with summary.as_default():
                  tf.summary.scalar('loss', total_loss, step=step + 1000*epoch)
                  tf.summary.scalar('learning rate', active_lr_float / 10** np.floor(np.log10(active_lr_float)), step=step + 1000*epoch)
                  tf.summary.scalar('accuracy', accuracy, step=step + + 1000*epoch)
                  tf.summary.scalar('Legal_prob', Legal_prob, step=step + 1000*epoch)
            total_steps += 1
            # Use carriage return to overwrite the current line
            print(
                f"Step: {step}, Lr: {(active_lr_float / 10** np.floor(np.log10(active_lr_float))):.1f} 10^{int(np.floor(np.log10(active_lr_float)))}, Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Legal_prob: {Legal_prob:.4f}, time : {(time.time() - timer):.1f}"
                ,end="\r")
            # summary.flush()
        # Write the results to the log file
        writer.writerow({"epoch": epoch, "total_loss" : float(total_loss), "accuracy" : float(accuracy), "Legal_prob" : float(Legal_prob)})
        print()  # Move to the next line after completing the epoch
        

#tf.print("loading weights")
#model.load_weights(f"model_2023-09-03_12-56-42_120.h5")
#tf.print("weights loaded")

#tf.print("going back to previous batches")
#for i in tqdm(range(120*1000)):
    #next(gen)
#train(1000, model)


