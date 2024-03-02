import tensorflow as tf

from models.lc0_az_policy_map import make_map



def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=64, kernel_size = (2,2)):
    print(prev_layer_input.shape)  
    up = tf.keras.layers.Conv2DTranspose(
                 n_filters,
                 kernel_size,
                 padding='valid')(prev_layer_input)  
    print(up.shape)  
    merge = tf.keras.layers.Concatenate(axis=3)([up, skip_layer_input])
    conv = tf.keras.layers.Conv2D(n_filters, 
                 kernel_size,  
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = tf.keras.layers.Conv2D(n_filters,
                 kernel_size, 
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv


def Uresidual_block(inputs, channels, name):
    res4_4 = tf.keras.layers.Conv2D(channels,
                                    5,
                                    use_bias=False,
                                    padding='valid',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    
                                    name=name + '/4_4/conv2d')(inputs)
    
    res4_4 = tf.keras.layers.Activation('relu')(
        batch_norm(res4_4, name + '/4_4/bn', scale=False))
    
    res2_2 = tf.keras.layers.Conv2D(channels,
                                    3,
                                    use_bias=False,
                                    padding='valid',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    
                                    name=name + '/2_2/conv2d')(res4_4)
    res2_2 = tf.keras.layers.Activation('relu')(
        batch_norm(res2_2, name + '/2_2/bn', scale=False))
    res1_1 = tf.keras.layers.Conv2D(channels,
                                    2,
                                    use_bias=False,
                                    padding='valid',
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=l2reg,
                                    
                                    name=name + '/1_1/conv2d')(res2_2)
    res1_1 = tf.keras.layers.Activation('relu')(
        batch_norm(res1_1, name + '/1_1/bn', scale=False))
    
    up_2_2 = DecoderMiniBlock(res1_1, res2_2, n_filters = channels, kernel_size=(2,2))
    up_4_4 = DecoderMiniBlock(up_2_2, res4_4, n_filters = channels, kernel_size=(3,3))
    up_8_8 = DecoderMiniBlock(up_4_4, inputs, n_filters = channels, kernel_size=(5,5))
    
    


    out = squeeze_excitation(batch_norm(up_8_8,
                                                    name + '/8_8_up/bn',
                                                    scale=True),
                                    channels,
                                    name=name + '/se')
    return tf.keras.layers.Activation('relu')(
        tf.keras.layers.add([inputs, out]))


def residual_model( channels, name):
    inputs = tf.keras.layers.Input(shape=(8,8,channels))
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
    out3 = tf.keras.layers.Activation('relu')(
        tf.keras.layers.add([inputs, out2]))
    conv3 = tf.keras.layers.Conv2D(channels-2,
                                3,
                                use_bias=False,
                                padding='same',
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=l2reg,
                                
                                name=name + '/3/conv2d')(out3)
    output = tf.keras.layers.Activation('relu')(
    batch_norm(conv3, name + '/3/bn', scale=False))
    return tf.keras.Model(inputs=inputs, outputs=output)

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
    




def create_body(inputs,num_Uresiduals=8, num_filters = 64,mini_res_channels = 64):
    flow = conv_block(inputs[0],
                        filter_size=3,
                        output_channels=num_filters,
                        name='input',
                        bn_scale=True)
    # mini_residual = residual_model(mini_res_channels,name='mini_residual')
    for i in range(num_Uresiduals):
        flow = Uresidual_block(flow,
                                num_filters,
                                name='residual_{}'.format(i + 1))
        #main,mini_res_input = flow[:,:,:,mini_res_channels-2:],flow[:,:,:,:mini_res_channels-2]
        #mini_res_output = mini_residual(tf.concat([mini_res_input,inputs[1]],axis = -1))
        #flow = tf.keras.layers.concatenate([main,mini_res_output],axis=-1)
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
    num_filters = params['num_filters']
    nb_residuals = params['num_residuals']
    mini_res_channels= params['mini_res_channels']
    input1 = tf.keras.layers.Input(shape=(8,8,nb_channels))
    input2 = tf.keras.layers.Input(shape=(8,8,2))
    x = create_body([input1,input2],num_Uresiduals=nb_residuals, num_filters = num_filters, mini_res_channels = mini_res_channels )

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
        
    model = tf.keras.Model(inputs=[input1,input2], outputs=h_fc1)

    return model