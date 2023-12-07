import tensorflow as tf


def create_simulator():
    x = tf.keras.Input((8,8,12))

    y = tf.keras.Input((1858))

    y_reshaped = tf.keras.layers.Dense((8*8*5))(y)
    y_reshaped = tf.keras.layers.Reshape((8,8,5))(y_reshaped)

    def small_block():
        inputs = tf.keras.Input((8,8,12+5))
        
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(inputs)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(flow)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        res = tf.keras.layers.Conv2D(1,(9,9),padding="same")(flow)
        res = tf.keras.layers.Activation("sigmoid")(res)
        return tf.keras.Model(inputs = inputs, outputs = res)


    flow = tf.concat([small_block()(tf.concat([x,y_reshaped],axis=-1)) for i in range(12)],axis=-1)
    
    output = flow
    
    return tf.keras.Model(inputs = [x,y], outputs = output)
