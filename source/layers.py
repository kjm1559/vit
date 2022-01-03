import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Reshape, Permute, Dropout, GlobalAveragePooling1D, Embedding
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
    def __init__(self, d_model, head=12):
        super(multiHeadAttentionLayer, self).__init__()
        self.head = head
        self.permute = Permute((2, 1, 3))
        self.re1 = Reshape((-1, self.head, d_model//self.head))
        self.re2 = Reshape((-1, d_model))
        self.linear = Dense(d_model)
                           

    def call(self, x, training):
        q, k, v = x
        # subspace header
        q_s = self.permute(self.re1(q))
        k_s = self.permute(self.re1(k))
        v_s = self.permute(self.re1(v))
        
        # combine head
        head = scaledDotProductAttentionLayer()([q_s, k_s, v_s], training)
        scaled_attention = self.permute(head)
        concat_attention = self.re2(self.permute(scaled_attention))
        multi_head = self.linear(concat_attention)
        return multi_head

class mlpLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super(mlpLayer, self).__init__()
        self.d1 = Dense(hidden_dim, activation=gelu)#, kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.d2 = Dense(output_dim)#, kernel_regularizer=tf.keras.regularizers.l2(0.1))
    def call(self, x, training):
        x = self.d1(x)
        return self.d2(x)

class transformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, head_num=12, h_dim=3072):
        super(transformerBlock, self).__init__()
        self.q_d = Dense(d_model)
        self.k_d = Dense(d_model)
        self.v_d = Dense(d_model)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.mlp = mlpLayer(h_dim, d_model)
        self.att = multiHeadAttentionLayer(d_model, head_num)
        
        self.drop = Dropout(0.1)
    
    def call(self, x, training):
        y = self.ln1(x)
        # multi head attention
        q = self.q_d(y) # query 
        k = self.k_d(y) # key
        v = self.v_d(y) # value
        y = self.att([q, k, v], training)
#         y = self.drop(y)

        # skip connection
        x = x + y

        # MLP layer
        y = self.ln2(x)
        y = self.mlp(y, training)
#         self.drop(y)
        
        # skip connection
        return x + y

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch+1, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch+1, delta=1)
        return self.proj(patch) + self.pos_embed(pos)
    
class visionTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, d_model=768, layer_num=12, head_num=12, h_dim=3072):
        super(visionTransformerLayer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_num = layer_num
        
        # learnabel class embedding
        self.class_emb = Dense(1)
        self.pos_emb = Embedding(input_dim=self.num_patches + 1, output_dim=768)
        
        self.patch_emb = [PatchEmbedding(self.num_patches, d_model) for i in range(layer_num)]
        
        self.per = Permute((2, 1))
#         self.class_emb = self.add_weight(shape=(1, 1, self.d_model),
#                                         initializer='random_normal',
#                                         trainable=True)
        
        # learnable position embedding
#         self.pos_emb = self.add_weight(shape=(1, 1, self.d_model), 
#                                       initializer='random_normal',
#                                       trainable=True)
        
        self.dense = Dense(d_model, activation='linear')
        self.t_layer = [transformerBlock(d_model, head_num, h_dim) for i in range(layer_num)]
        
    def call(self, x, training):
        # feature extraction
        batch_size = tf.shape(x)[0]
        
        # resize image
        x = tf.image.resize(x, [self.image_size, self.image_size])
        
        # extract patch
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = Reshape((self.num_patches, -1))(patches)
        print('patches:', patches.shape, self.patch_size, self.num_patches)
        x = self.dense(patches)
        print(f'patches: {x.shape}')
        
        pos = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        class_emb = self.per(self.class_emb(self.per(x)))
        pos_emb = self.pos_emb(pos)#self.per(self.pos_emb(self.per(x)))
#         class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        print('class_emb:', pos_emb.shape)
        x = tf.concat([class_emb, x], axis=1)
        x = x + pos_emb
        
        # transformer block
        for i in range(self.layer_num):
            x = self.patch_emb[i](x)
            x = self.t_layer[i](x, training)
        
        return x
        

def visionTransformer(input_dim, output_dim, image_size=32, patch_size=8, d_model=768, layer_num=12, head_num=12, h_dim=3072):
    inputs = tf.keras.Input(shape=input_dim)
    ViT_layer = visionTransformerLayer(image_size, patch_size, d_model, layer_num, head_num, h_dim)
    y = ViT_layer(inputs)
    print('y :', y.shape)
#     y = Dense(output_dim, activation=gelu)(y[:, 0])
    y = GlobalAveragePooling1D()(y)
    outputs = Dense(output_dim, activation='softmax')(y)
    print(outputs.shape)
    return tf.keras.Model(inputs, outputs, name='vit'), ViT_layer
    
    