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
    tf.config.experimental_run_functions_eagerly(True)
    reduce_cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = reduce_cifar10.load_data()
    
    print(X_train.shape)
    
    y_train = np.squeeze(np.eye(10)[y_train])
    y_test = np.squeeze(np.eye(10)[y_test])
    
    print(y_train.shape, X_train.shape)

    #normalization
    X_train = X_train/255
    X_test = X_test/255
        
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model = visionTransformer(X_train.shape[1:], y_train.shape[-1])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(3e-4), metrics=['acc'])
    print('start training ...')
    model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=400, callbacks=[es])
    print('start evaluation ...')
    model.evaluate(X_test, y_test)
    