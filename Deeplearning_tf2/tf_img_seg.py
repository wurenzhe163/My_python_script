import tensorflow.keras.backend as K
import numpy as np
import os
import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten,Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l2
import PIL
"""
深度学习运算，Unet为对应模型，可替换
"""


# 变换image，内嵌
def random_jitter(image):
    image = tf.image.random_brightness(image, 0.5)  # 随机改变亮度,0.5为50%
    image = tf.image.random_contrast(image, lower=0.2, upper=1)  # 对比度，0，1为界限
    return image


# 固定变换，内嵌
def image_flip(image, left_right=0, up_down=0):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def load_pro_(path, kernel):
    item = tf.strings.split(path, " ")  # 同时赋予标签和image
    img_raw = tf.io.read_file(item[0])
    img_tensor = tf.image.decode_png(img_raw, channels=3)
    img_tensor = tf.image.resize(img_tensor, kernel)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = random_jitter(img_tensor)
    img = img_tensor / 255.

    label_raw = tf.io.read_file(item[1])
    label_tensor = tf.image.decode_png(label_raw, channels=1)
    label_tensor = tf.image.resize(label_tensor, kernel)
    label_tensor = tf.cast(label_tensor, tf.float32)

    train = tf.concat([img, label_tensor], axis=2)
    train = image_flip(train)
    img = train[:, :, 0:3]
    label_tensor = train[:, :, 3:4]
    return img, label_tensor

def load_pro_2(path,kernel):

    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_png(img_raw, channels=3)
    img_tensor = tf.image.resize(img_tensor, kernel)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255.

    return img


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))#[:,:,0:1]
        plt.axis('off')
    plt.show()


def UNet(img_h, img_w):
    def bn_act(x, act=True, activation='relu'):
        'BN层+RELU层，使用act=True or False 开关激活'
        x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation(activation)(x)
        return x

    inputs = Input((img_h, img_w, 3))
    e0 = Conv2D(32, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(inputs)  # 512
    e0 = tf.keras.layers.BatchNormalization()(e0)
    e0 = Conv2D(32, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(e0)
    e0 = tf.keras.layers.BatchNormalization()(e0)

    pool1 = MaxPool2D(pool_size=(2, 2))(e0)

    e1 = Conv2D(64, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(pool1)  # 256
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1 = Conv2D(64, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(e1)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    pool2 = MaxPool2D(pool_size=(2, 2))(e1)

    e2 = Conv2D(128, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(pool2)  # 128
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = Conv2D(128, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(e2)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    pool3 = MaxPool2D(pool_size=(2, 2))(e2)

    e3 = Conv2D(256, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(pool3)  # 64
    e3 = tf.keras.layers.BatchNormalization()(e3)
    e3 = Conv2D(256, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(e3)
    e3 = tf.keras.layers.BatchNormalization()(e3)
    pool4 = MaxPool2D(pool_size=(2, 2))(e3)

    brige = Conv2D(512, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(
        pool4)  # 32
    brige = tf.keras.layers.BatchNormalization()(brige)
    brige = Conv2D(512, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(brige)
    brige = tf.keras.layers.BatchNormalization()(brige)
    # -------------------------------------------------------------------------
    up3 = UpSampling2D((2, 2))(brige)
    merge0 = Concatenate()([up3, e3])
    conv0 = Conv2D(256, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(
        merge0)  # 64
    conv0 = tf.keras.layers.BatchNormalization()(conv0)
    conv0 = Conv2D(256, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(conv0)
    conv0 = tf.keras.layers.BatchNormalization()(conv0)

    up2 = UpSampling2D((2, 2))(conv0)
    merge1 = Concatenate()([up2, e2])
    conv1 = Conv2D(128, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(
        merge1)  # 128
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = Conv2D(128, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

    up1 = UpSampling2D((2, 2))(conv1)
    merge2 = Concatenate()([up1, e1])
    conv2 = Conv2D(64, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(
        merge2)  # 256
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    up0 = UpSampling2D((2, 2))(conv2)
    merge3 = Concatenate()([up0, e0])
    conv3 = Conv2D(32, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(merge3)  # 512
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same', strides=1, kernel_initializer='he_normal')(conv3)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet
def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1-pt_1), gamma)

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def model_loadweight(model,pretrained_model_path,load_pretrained_model=False):

    if load_pretrained_model:
        try:
            model.load_weights(pretrained_model_path)
            print('pre-trained model loaded!')
        except OSError:
            print('You need to run the model and load the trained model')

def model_fit_train(model,TRAIN_LENGTH,TEST_LENGTH,BATCH_SIZE,EPOCHS,train_dataset,test_dataset):

    import os, datetime
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # 时间存储路径
    tensorborad_callback = tf.keras.callbacks.TensorBoard(log_dir,
                                                          histogram_freq=0)  # tensorboard回调函数 histogram_freq=0  不计算模型各层的激活度和权重直方图的频率
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=1, mode='auto',
                                                     min_delta=0.0001, cooldown=0, min_lr=0)
    checkpoint_path = "Weight/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, save_freq=10)


    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE
    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              shuffle=True,
                              steps_per_epoch=STEPS_PER_EPOCH,       #在宣布一个epoch结束并开始下一个epoch之前的总步骤数(批量样本)
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              initial_epoch = 0,
                              callbacks=[reduce_lr,tensorborad_callback,cp_callback]
                              )
def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
def save_image(img,pathname,name='road'):

    """保存,PIL支持uint8，但是有的图像压缩成uint8会出错"""
    make_dir(pathname)
    #for i in range(len(a_append)):
    A = PIL.Image.fromarray(img)
    s = pathname+'\\'+name+'.png'
    A.save(s,quality=95)



# 以前传统使用
def prediction(model,seg_mark,dataset=None, pathname='C:\\Users\\SAR\\Desktop\\work_Photovoltaic\\20201217\\pre', name='pre'):

    make_dir(pathname)
    def create_mask(pred_mask):
        pred_mask = np.argmax(pred_mask, axis=-1)
        #pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]
    if seg_mark ==1:
        for i, image in enumerate(dataset):
            pred_mask = create_mask(model.predict(image[tf.newaxis, ...]))
            pred_mask = pred_mask.astype(np.uint8)
            if i % 100 == 0 :
                print(i, end='\n')
            name0 = name + f'{i:04d}'
            save_image(pred_mask, pathname, name=name0)
    elif seg_mark == 2:
        for i, image in enumerate(dataset):
            pred_mask = model.predict(image[tf.newaxis, ...])
            pred_mask = np.squeeze(pred_mask)
            pred_mask[pred_mask > 0.1] = 1
            pred_mask[pred_mask <= 0.1] = 0
            pred_mask = pred_mask.astype(np.uint8)
            if i % 100 == 0:
                print(i, end='\n')
            name0 = name + f'{i:04d}'
            save_image(pred_mask, pathname, name=name0)
        # pred_mask = model.predict(image[tf.newaxis, ...]).reshape(256, 256)
        # pred_mask[pred_mask > 0.1] = 255
        # pred_mask[pred_mask <= 0.1] = 0


def evaluate_1(model,dataset=None):
    str__,acc__,loss__ = [],[],[]
    for i, (image,label) in enumerate(dataset):
        pred_mask = model.evaluate(image,label)
        loss = float("%.4f" % pred_mask[0])
        accuracy = float("%.4f" % pred_mask[1])

        acc__.append(accuracy)
        loss__.append(loss)

        str__.append(f'{i:04d}'+'   '+str(accuracy)+'   '+str(loss))
    with open('paramete.txt','w') as f:
        f.writelines('FID      ACC     LOSS\n')
        for each_str in str__:
            f.writelines(each_str+'\n')
        f.writelines('       '+'   '+("%.4f"%np.mean(acc__))+'   '+("%.4f"%np.mean(loss__)))

def evaluate_2(model,dataset=None):
    str__,iou__,acc__,loss__,tversky__ = [],[],[],[],[]
    for i, (image,label) in enumerate(dataset):
        pred_mask = model.evaluate(image,label)
        loss = float("%.4f" % pred_mask[0])
        tversky = float("%.4f" % pred_mask[1])
        accuracy = float("%.4f" % pred_mask[2])
        iou = float("%.4f" % pred_mask[3])
        iou__.append(iou)
        acc__.append(accuracy)
        loss__.append(loss)
        tversky__.append(tversky)
        str__.append(f'{i:04d}'+'   '+str(iou)+'   '+str(accuracy)+'   '+str(loss)+'    '+str(tversky))
    with open('paramete.txt','w') as f:
        f.writelines('FID         IOU      ACC     LOSS   Tversky\n')
        for each_str in str__:
            f.writelines(each_str+'\n')
        f.writelines('       '+'   '+("%.4f"%np.mean(iou__))+'   '+("%.4f"%np.mean(acc__))+'   '+("%.4f"%np.mean(loss__))+'    '+("%.4f"%np.mean(tversky__)))
