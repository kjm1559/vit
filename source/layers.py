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
    def __init__(self, d_model, head=4):
        super(multiHeadAttentionLayer, self).__init__()
        self.head = head
        self.permute = Permute((2, 1, 3))
        self.re1 = Reshape((-1, self.head, d_model//self.head))
        self.re2 = Reshape((-1, d_model))
                           

    def call(self, x, training):
        q, k, v = x
        # subspace header
        q_s = self.permute(self.re1(q))
        k_s = self.permute(self.re1(k))
        v_s = self.permute(self.re1(v))
        
        # combine head
        head = scaledDotProductAttentionLayer()([q_s, k_s, v_s], training)
        head = self.re2(self.permute(head))
        multi_head = linear(head)
        return multi_head

class mlpLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super(mlpLayer, self).__init__()
        self.d1 = Dense(hidden_dim, activation=gelu)
        self.d2 = Dense(output_dim)
    def call(self, x, training):
        x = self.d1(x)
        return self.d2(x)

class transformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, h_dim=128):
        super(transformerBlock, self).__init__()
        self.q_d = Dense(d_model)
        self.k_d = Dense(d_model)
        self.v_d = Dense(d_model)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.mlp = mlpLayer(h_dim, d_model)
        self.att = multiHeadAttentionLayer(d_model)
    
    def call(self, x, training):
        y = self.ln1(x)

        # multi head attention
        q = self.q_d(y) # query 
        k = self.k_d(y) # key
        v = self.v_d(y) # value
        y = self.att([q, k, v], training)

        # skip connection
        x = x + y

        # MLP layer
        y = self.ln2(x)
        y = self.mlp(y, training)

        # skip connection
        return x + y        

    
class visionTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, d_model=64, layer_num=8):
        super(visionTransformerLayer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.patch_size = patch_size
        self.layer_num = layer_num
        
        # learnabel class embedding
        self.class_emb = self.add_weight(shape=(1, 1, self.d_model),
                                        initializer='random_normal',
                                        trainable=True)
        
        # learnable position embedding
        self.pos_emb = self.add_weight(shape=(1, self.num_patches + 1, self.d_model), 
                                      initializer='random_normal',
                                      trainable=True)
        
        self.dense = Dense(d_model)
        self.t_layer = [transformerBlock(d_model) for i in range(layer_num)]
        
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
        
        x = self.dense(patches)
        
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb
        
        # transformer block
        for i in range(self.layer_num):
            x = self.t_layer[i](x, training)
        
        return x
        

def visionTransformer(input_dim, output_dim, image_size=32, patch_size=4):
    inputs = tf.keras.Input(shape=input_dim)
    y = visionTransformerLayer(image_size, patch_size)(inputs)
    print('y :', y.shape)
    y = Dense(output_dim, activation=gelu)(y[:, 0])
    outputs = Dense(output_dim, activation='softmax')(y)
    print(outputs.shape)
    return tf.keras.Model(inputs, outputs, name='vit')
    
    