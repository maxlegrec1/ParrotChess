from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout, Embedding
import tensorflow as tf

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(h, d_k)
        self.multihead_attention2 = tf.keras.layers.MultiHeadAttention(h, d_k)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, encoder_output, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x,training=training)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)

        multihead_output2 = self.multihead_attention2(addnorm_output, encoder_output, encoder_output, training=training)
        # Expected output shape = (batch_size, sequence_length, d_model)

        addnorm_output2 = self.add_norm1(x, multihead_output2)
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Decoder(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
    def call(self, input_sentence, padding_mask, training):

        x = input_sentence
        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x


if __name__ == "__main__":
    # Define the hyperparameters

    h = 8
    d_k = 64
    d_v = 64
    d_model = 1024
    d_ff = 2048
    n = 6
    rate = 0.0

    # Create an encoder
    encoder = Decoder( h, d_k, d_v, d_model, d_ff, n, rate)

    input_sentence = tf.random.uniform((2, 1958,1024),maxval = vocab_size, dtype=tf.int32)
    input_sentence = tf.cast(input_sentence,tf.float32)
    encoder_output = tf.random.uniform((2,64,1024),maxval = 2, dtype=tf.int32)
    encoder_output = tf.cast(encoder_output, tf.float32)

    # Pass the input through the encoder
    output = encoder(input_sentence, encoder_output, training=True)

    # Print the shape of the output
    print(output.shape)  # Expected output shape: (batch_size, sequence_length, d_model)

    print("transformer can work with sentences of different lengths !")