
import importlib
from config.string_to_elements import from_string_to_fun
import tensorflow as tf
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
def main():
    converter = from_string_to_fun()
    config_name = 'default_config'
    params = importlib.import_module('config.'+ config_name).parameters()
    arguments = params['shared_parameters']
    training_args = params['training_args']
    print(params)
    data_generator = converter[params['data_generator']](arguments)
    model = converter[params['model']](arguments)
    trainer = converter[params['trainer']](arguments)
    lr_start =  arguments['lr_start']
    
    active_lr = tf.Variable(lr_start, dtype=tf.float32,trainable=False)
    optimizer = tf.keras.optimizers.SGD(
                    learning_rate=active_lr,
                    momentum=0.9,
                    nesterov=True)
    model.compile(optimizer = optimizer)

    model.summary()
    trainer(data_generator, model, **training_args)
    
if __name__ == '__main__':
    main()