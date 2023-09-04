import tensorflow as tf
import numpy as np
from tqdm import tqdm

def residual(x,num_filter):
    skip= x
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([skip, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def create_A0(num_residuals,out_channels):
    input = tf.keras.Input(shape=(8, 8, 79))
    x = input
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    for _ in range(num_residuals):
        x = residual(x, 256)
    
    #convolution to get the out_channels
    output = tf.keras.layers.Conv2D(12, (3, 3), padding='same',activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

generator = create_A0(40,12)

generator.summary()
generator_optimizer = tf.keras.optimizers.Adam(1e-6)

def create_discriminator(input_shape=(8, 8, 24)):

    def expansion_block(x, t, filters, block_id):
        prefix = 'block_{}_'.format(block_id)
        total_filters = t * filters
        x = tf.keras.layers.Conv2D(
            total_filters, 1, padding='same', use_bias=False, name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = tf.keras.layers.ReLU(6, name=prefix + 'expand_relu')(x)
        return x

    def depthwise_block(x, stride, block_id):
        prefix = 'block_{}_'.format(block_id)
        x = tf.keras.layers.DepthwiseConv2D(3, strides=(
            stride, stride), padding='same', use_bias=False, name=prefix + 'depthwise_conv')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + 'dw_bn')(x)
        x = tf.keras.layers.ReLU(6, name=prefix + 'dw_relu')(x)
        return x

    def projection_block(x, out_channels, block_id):
        prefix = 'block_{}_'.format(block_id)
        x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,
                                   padding='same', use_bias=False, name=prefix + 'compress')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + 'compress_bn')(x)
        return x

    def Bottleneck(x, t, filters, out_channels, stride, block_id):
        y = expansion_block(x, t, filters, block_id)
        y = depthwise_block(y, stride, block_id)
        y = projection_block(y, out_channels, block_id)
        if y.shape[-1] == x.shape[-1]:
            y = tf.keras.layers.add([x, y])
        return y

    # Input layer
    input_layer = tf.keras.layers.Input(input_shape)
    # Convolution layers
    x = tf.keras.layers.Conv2D(32, 3, strides=(
        2, 2), padding='same', use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)
    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=16, block_id=1)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=2, block_id=2)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=3)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=2, block_id=4)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=5)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=6)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=2, block_id=7)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=8)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=9)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=10)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=11)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=12)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=13)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=2, block_id=14)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=15)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=16)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=320, stride=1, block_id=17)
    x = tf.keras.layers.Conv2D(filters=1280, kernel_size=1,
                               padding='same', use_bias=False, name='last_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='last_bn')(x)
    x = tf.keras.layers.ReLU(6, name='last_relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pool')(x)

    output = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model(input_layer, output)
    return model

discriminator = create_discriminator((8,8,90))

discriminator.summary()
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(batch):

    x,y,x_without_noise = batch
    #x = coup before + legal mask + color + elo, y = coup after
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_outputs = generator(x, training=True)
        real_input_disc = tf.concat([x_without_noise, y], axis=-1)
        fake_input_disc = tf.concat([x_without_noise, gen_outputs], axis=-1)
        real_output = discriminator(real_input_disc, training=True)
        fake_output = discriminator(fake_input_disc, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)



        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
            cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))
        
def train(num_steps,gen_batch):

    for step in tqdm(range(num_steps)):
        batch = next(gen_batch)
        train_step(batch)
        if step % 100 == 0:
            print('.', end='', flush=True)
        if step % 100 == 0:
            #evaluate performance of generator on a batch
            #x = coup before + legal mask + color + elo, y = coup after, disc_true = coup before
            x,y,x_without_noise = next(gen_batch)
            gen_outputs = generator(x, training=False)
            fake_input_disc = tf.concat([x_without_noise, gen_outputs], axis=-1)
            fake_output = discriminator(fake_input_disc, training=False)
            print('discriminator output on fake images: ',fake_output)
            print('generator loss: ',cross_entropy(tf.ones_like(fake_output), fake_output))

        if step % 1000 == 0:
            generator.save_weights('./checkpoints/generator_{}.h5'.format(step))
            discriminator.save_weights('./checkpoints/discriminator_{}.h5'.format(step))


#write if name == main
if __name__ == '__main__':
    from refactored import generate_batch
    gen = generate_batch()
    num_steps = 10000000
    train(num_steps,gen)

