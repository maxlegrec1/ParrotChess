import tensorflow as tf
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
import csv
import wandb
import os



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




#custom training loop
@tf.function
def train_step(batch,model,gradient_acc_steps = 1):
    x,y_true,value_true = batch
    #all non negative values should be 1
    mask = tf.cast(tf.math.greater_equal(y_true,0),tf.float32) 
    y_true = tf.nn.relu(y_true)
    if gradient_acc_steps==1:
        with tf.GradientTape() as tape:
                #tf.print(i)
                y_all = model(x)
                y_pred = y_all['policy']
                value_pred = y_all['value']
                #loss = tf.keras.losses.categorical_crossentropy(tf.stop_gradient(y_true),y_pred)
                loss1 =tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),logits=y_pred)
                loss2 =tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(value_true),logits=value_pred)
                
                loss = loss1 + loss2

                gradients = tape.gradient(loss, model.trainable_variables)
        
        gradients, _ = tf.clip_by_global_norm(gradients, 10000)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        lm = tf.reduce_sum(mask*tf.keras.layers.Softmax()(y_pred),axis=-1)

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32),axis=-1)

        value_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(value_true,axis=-1),tf.argmax(value_pred,axis=-1)),tf.float32),axis=-1)

    else:

        for i in range(gradient_acc_steps):
            with tf.GradientTape() as tape:
                #tf.print(i)
                y_pred = model([x[0][i*y_true.shape[0]//gradient_acc_steps:(i+1)*y_true.shape[0]//gradient_acc_steps],x[1][i*y_true.shape[0]//gradient_acc_steps:(i+1)*y_true.shape[0]//gradient_acc_steps]])
                #loss = tf.keras.losses.categorical_crossentropy(tf.stop_gradient(y_true),y_pred)
                loss =tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true[i*y_true.shape[0]//gradient_acc_steps:(i+1)*y_true.shape[0]//gradient_acc_steps]),logits=y_pred)/gradient_acc_steps
                
                gradients = tape.gradient(loss, model.trainable_variables)
                
                accum_ops = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]

        accum_vars, _ = tf.clip_by_global_norm(accum_vars, 10000)

        model.optimizer.apply_gradients(zip(accum_vars, model.trainable_variables))

        #tf.print(mask.dtype,y_pred.dtype)
        lm = tf.reduce_sum(mask[i*y_true.shape[0]//gradient_acc_steps:(i+1)*y_true.shape[0]//gradient_acc_steps]*tf.keras.layers.Softmax()(y_pred),axis=-1)

        #tf.print(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true[i*y_true.shape[0]//gradient_acc_steps:(i+1)*y_true.shape[0]//gradient_acc_steps],axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32),axis=-1)

    return loss,lm,acc,value_acc


def trainer(params):
    def aux(*args, **kwargs):
        return train(*args,**kwargs, lr_start=params['lr_start'], lr = params['lr'])
    return aux




def train(gen, model, num_step, lr_start ,lr, warmup_steps, num_report_steps, resume_id, start_from, total_num_steps):
    if resume_id == None:
        id = wandb.util.generate_id()
        wandb.init(project='owt', id= id, resume = 'allow')
    else:
        id = resume_id
        wandb.init(project='owt', id= id, resume = 'allow')
        model.load_weights(wandb.restore(f"model_last_{id}.h5").name)
    #create a log file where we will store the results. It shall be named after the current date and time
    print("id : " ,id)
    total_steps = start_from*num_report_steps
    best_model_acc = 0
    for epoch in range(start_from,total_num_steps//num_report_steps):
        timer = time.time()
        total_loss = 0
        Legal_prob = 0
        accuracy = 0
        value = 0
        if epoch%40==0 and epoch!=0:
            #save weights
            model.save_weights(f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.h5")

        for step in range(num_step):
            batch = gen.get_batch()
            active_lr_float = (lr_start + (lr - lr_start) * min(1, (total_steps + 1) / warmup_steps))
            model.optimizer.lr.assign(active_lr_float)



            


            loss,lm,acc,value_acc = train_step(batch, model)
            del batch
            loss = tf.reduce_mean(loss)
            lm = tf.reduce_mean(lm)
            total_loss = total_loss / (step + 1) * step + loss / (step + 1)
            Legal_prob = Legal_prob / (step + 1) * step + lm / (step + 1)

            accuracy = accuracy / (step + 1) * step + acc / (step + 1)

            value = value / (step + 1) * step + value_acc / (step + 1)

            total_steps += 1

            print(
                f"Step: {total_steps}, Lr: {(active_lr_float / 10** np.floor(np.log10(active_lr_float))):.1f} 10^{int(np.floor(np.log10(active_lr_float)))}, Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Legal_prob: {Legal_prob:.4f}, Value: {value:.4f}, time : {(time.time() - timer):.1f}"
                ,end="\r")
        print()

        wandb.log({"train/loss": total_loss, "accuracy": accuracy, "Legal_prob": Legal_prob, "lr": active_lr_float, "iter": total_steps+1, "value_accuracy": value})
        if accuracy >= best_model_acc:
            best_model_acc = accuracy
            model.save_weights(os.path.join(wandb.run.dir,f"model_best_{id}.h5"))
        model.save_weights(os.path.join(wandb.run.dir,f"model_last_{id}.h5"))

