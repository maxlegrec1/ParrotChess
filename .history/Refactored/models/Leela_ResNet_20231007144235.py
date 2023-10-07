import tensorflow as tf

from models.lc0_az_policy_map import make_map

def residual_block(inputs, channels, name):
    conv1 = tf.keras.layers.Conv2D(channels,
                                    3,
                                    use_bias=False,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    
                                    name=name + '/1/conv2d')(inputs)
    out1 = tf.keras.layers.Activation('relu')(
        batch_norm(conv1, name + '/1/bn', scale=False))
    conv2 = tf.keras.layers.Conv2D(channels,
                                    3,
                                    use_bias=False,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    
                                    name=name + '/2/conv2d')(out1)

    out2 = squeeze_excitation(batch_norm(conv2,
                                                    name + '/2/bn',
                                                    scale=True),
                                    channels,
                                    name=name + '/se')
    return tf.keras.layers.Activation('relu')(
        tf.keras.layers.add([inputs, out2]))


def batch_norm(input, name, scale=False):
    return tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        axis=1,
        center=True,
        scale=scale,
        virtual_batch_size=None,
        name=name)(input)


def squeeze_excitation(inputs, channels, name):
    assert channels % 2 == 0

    pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    squeezed = tf.keras.layers.Activation('relu')(
        tf.keras.layers.Dense(channels // 2,
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=l2reg,
                                name=name + '/se/dense1')(pooled))
    excited = tf.keras.layers.Dense(2 * channels,
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    name=name + '/se/dense2')(squeezed)
    return ApplySqueezeExcitation()([inputs, excited])

l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))

class ApplySqueezeExcitation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, 1, 1, self.reshape_size]),
                                 2,
                                 axis=-1)
        return tf.nn.sigmoid(gammas) * x + betas
    




def create_body(inputs,num_residuals=16, num_filters = 64):
    flow = conv_block(inputs,
                        filter_size=3,
                        output_channels=num_filters,
                        name='input',
                        bn_scale=True)
    for i in range(num_residuals):
        flow = residual_block(flow,
                                num_filters,
                                name='residual_{}'.format(i + 1))
    return flow

def conv_block(inputs,
                filter_size,
                output_channels,
                name,
                bn_scale=False):
    conv = tf.keras.layers.Conv2D(output_channels,
                                    filter_size,
                                    use_bias=False,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    name=name + '/conv2d')(inputs)
    return tf.keras.layers.Activation('relu')(
        batch_norm(conv, name=name + '/bn', scale=bn_scale))

def create_model(params):
    nb_channels = params['num_channels']
    nb_filters = params['num_filters']
    nb_residuals = params['num_residuals']
    input = tf.keras.layers.Input(shape=(8,8,nb_channels))

    x = create_body(input,num_residuals=nb_residuals, nb_filters = nb_filters )

    conv_pol = conv_block(x,
                                filter_size=3,
                                output_channels=64,
                                name='policy1')
    conv_pol2 = tf.keras.layers.Conv2D(
        80,
        3,
        use_bias=True,
        padding='same',
        kernel_initializer='glorot_normal',
        kernel_regularizer=l2reg,
        bias_regularizer=l2reg,
        name='policy')(conv_pol)
    #h_fc1 = ApplyPolicyMap()(conv_pol2)
    h_fc1 = tf.reshape(conv_pol2, [-1, 80 * 8 * 8])
    mat = tf.constant(make_map())
    h_fc1=tf.matmul(h_fc1, tf.cast(mat, tf.float32))
    #h_fc1 = tf.keras.layers.Softmax()(h_fc1)

    class ApplyPolicyMap(tf.keras.layers.Layer):

        def __init__(self, **kwargs):
            super(ApplyPolicyMap, self).__init__(**kwargs)
            self.fc1 = tf.constant(make_map())

        def call(self, inputs):
            h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
            return tf.matmul(h_conv_pol_flat,
                            tf.cast(self.fc1, h_conv_pol_flat.dtype))
        
    model = tf.keras.Model(inputs=input, outputs=h_fc1)

    return model