

path = 'F:\GitRefactored\ParrotChess\Refactored\logs\log_2023-10-16_20-30-40.csv'
name = '2023-10-16_20-30-40'

import pandas as pd
import tensorflow as tf

df = pd.read_csv(path,delimiter = ',')
print(df)

writer = tf.summary.create_file_writer(f"tensorboard/logdir_{name}")

for index,row in df.iterrows():
    with writer.as_default():
        tf.summary.scalar('loss',row['total_loss'],step = (index+1)*1000)
        tf.summary.scalar('Legal_prob',row['Legal_prob'],step = (index+1)*1000)
        tf.summary.scalar('accuracy',row['accuracy'],step = (index+1)*1000)
        tf.summary.scalar('learning rate',10**(-4-int(index/100)),step = (index+1)*1000)