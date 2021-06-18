import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, LayerNormalization, Conv2D, Flatten, Reshape, Permute
from tensorflow.keras.activations import softmax, linear
import tensorflow.keras.backend as K
import numpy as np

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

class scaledDotProductAttentionLayer(tf.keras.layers.Layer):
    def call(self, x, training):
        q, k, v = x
        qk = tf.matmul(q, k, transpose_b=True)/K.sqrt(tf.cast(K.shape(k)[-1], tf.float32))
        return tf.matmul(softmax(qk, axis=-1), v)
    

class multiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, head=4):
        super(multiHeadAttentionLayer, self).__init__()
        self.head = head

    def call(self, x, training):
        q, k, v = x
        # subspace header
        q_s = Permute((2, 1, 3))(Reshape((-1, self.head, q.shape[-1]//self.head))(q))
        k_s = Permute((2, 1, 3))(Reshape((-1, self.head, q.shape[-1]//self.head))(k))
        v_s = Permute((2, 1, 3))(Reshape((-1, self.head, q.shape[-1]//self.head))(v))
        
        # combine head
        head = scaledDotProductAttentionLayer()([q_s, k_s, v_s])
        head = Reshape((-1, q.shape[-1]))(Permute((2, 1, 3))(head))
        multi_head = linear(head)
        return multi_head
        

def mlp_layer(x, hidden_dim, output_dim):
    x = Dense(hidden_dim, activation=gelu)(x)
    output = Dense(output_dim)(x)
    return output

        
def transformerBlock(x, d_model, h_dim=128):
    y = LayerNormalization()(x)
    
    # multi head attention
    q = Dense(d_model)(y) # query 
    k = Dense(d_model)(y) # key
    v = Dense(d_model)(y) # value
    y = multiHeadAttentionLayer()([q, k, v])
    
    # skip connection
    x = x + y
    
    # MLP layer
    y = LayerNormalization()(x)
    y = mlp_layer(y, h_dim, d_model)
    
    # skip connection
    return x + y

    
class visionTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, d_model=64, layer_num=8):
        super(visionTransformerLayer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.patch_size = patch_size
        self.layer_num = layer_num
        
    def build(self, input_shape):
        # learnabel class embedding
        self.class_emb = self.add_weight(shape=(1, 1, self.d_model),
                                        initializer='random_normal',
                                        trainable=True)
        
        # learnable position embedding
        self.pos_emb = self.add_weight(shape=(1, self.num_patches + 1, self.d_model), 
                                      initializer='random_normal',
                                      trainable=True)
        
    def call(self, x, training):
        # feature extraction
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = Reshape((-1, self.patch_size**2 * x.shape[-1]))(patches)
        
        x = Dense(self.d_model)(patches)
        
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb
        
        # transformer block
        for i in range(self.layer_num):
            x = transformerBlock(x, self.d_model)
        
        return x
        

def visionTransformer(input_dim, output_dim, image_size=32, patch_size=16):
    inputs = tf.keras.Input(shape=input_dim)
    y = visionTransformerLayer(image_size, patch_size)(inputs)
    y = Dense(output_dim, activation=gelu)(y[:, 0])
    outputs = Dense(output_dim, activation='softmax')(y)
    return tf.keras.Model(inputs, outputs, name='vit')
    
    