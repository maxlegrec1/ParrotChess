
import importlib
from config.string_to_elements import from_string_to_fun
import tensorflow as tf
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
def main():
    
    GPU_ID = 0
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[GPU_ID], 'GPU')
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[GPU_ID], True)
    
    converter = from_string_to_fun()
    config_name = 'default_config'
    params = importlib.import_module('config.'+ config_name).parameters()
    print(params)

    arguments = params['shared_parameters']
    training_args = params['training_args']

    data_generator = converter[params['data_generator']](arguments)
    model = converter[params['model']](arguments)
    trainer = converter[params['trainer']](arguments)

    #learning rate will be set later on
    model.compile(optimizer = tf.keras.optimizers.Nadam(beta_1=0.9,beta_2=0.98))


    model.summary()
    trainer(data_generator, model, **training_args)
    
if __name__ == '__main__':
    main()