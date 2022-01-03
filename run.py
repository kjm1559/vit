import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
import numpy as np
from source.layers import visionTransformer

def patch_spliter(image, w, h):
    result = []
    for i in range(int(image.shape[0]/w)):
        for j in range(int(image.shape[1]/h)):
            result.append(image[i*h:(i+1)*h, j*w:(j+1)*h])
    return np.array(result)

def batch_patch_spliter(images, w, h):
    result = []
    for im in images:
        result.append(patch_spliter(im, w, h))
    return np.array(result)

if __name__ == '__main__':
#     tf.config.experimental_run_functions_eagerly(True)
    # pretrain
    reduce_cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = reduce_cifar10.load_data()
    
    print(X_train.shape)
    
    y_train = np.squeeze(np.eye(10)[y_train])
    y_test = np.squeeze(np.eye(10)[y_test])
    
    print(y_train.shape, X_train.shape, y_train.shape)

    #normalization
    X_train = X_train/255
    X_test = X_test/255
        
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=42, restore_best_weights=True)
    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=40, t_mul=2.0, m_mul=0.9, alpha=1e-2)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)
    
    # model is base/16
    model, header = visionTransformer(X_train.shape[1:], y_train.shape[-1])
#     model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(3e-4), metrics=['acc'])
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['acc'])
    print('start training ...')
    model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=20, callbacks=[es, ls])
    print('start evaluation ...')
    model.evaluate(X_test, y_test)
    
    model.save_weights('cifar10.h5')   
    
    # fine tune
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(X_train.shape)
    
    y_train = np.squeeze(np.eye(10)[y_train])
    y_test = np.squeeze(np.eye(10)[y_test])
    
    print(y_train.shape, X_train.shape, y_train.shape)
    
    X_train = np.concatenate([np.expand_dims(X_train, axis=-1)] * 3, axis=-1)
    X_test = np.concatenate([np.expand_dims(X_test, axis=-1)] * 3, axis=-1)

    #normalization
    X_train = X_train/255
    X_test = X_test/255
    
    # mnist
    model, header = visionTransformer(X_train.shape[1:], y_train.shape[-1])
    model.load_weights('cifar10.h5')
    # reset last layer's weight
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    for i in [-2, -1]:
        layer_new_weights = []
        for layer_weights in model.layers[i].get_weights():
            weights = initializer(np.shape(layer_weights))
            layer_new_weights.append(weights)
        model.layers[i].set_weights(layer_new_weights)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['acc'])
    print('start training ...')
    model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=20, callbacks=[es, ls])
    print('start evaluation ...')
    model.evaluate(X_test, y_test)
    