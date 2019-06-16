import keras
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from sklearn.metrics import log_loss 
from scale_layer import Scale
import sys
sys.setrecursionlimit(3000)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    
    x = Conv2D(nb_filter1, (1,1), name = conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=scale_name_base + '2a_relu')(x)
    
    x = ZeroPadding2D((1,1), name=scale_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=scale_name_base + '2b_relu')(x)
    
    x = Conv2D(nb_filter3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    
    x = Conv2D(nb_filter1, (1,1), strides = strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon = eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    
    x = ZeroPadding2D((1,1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon = eps, axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    
    x = Conv2D(nb_filter3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
    
    shortcut = Conv2D(nb_filter3, (1,1), strides=strides, name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)
    
    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    
    return x

def resnet152_model(img_rows, img_cols, color_type=1, num_classes=None):
    eps = 1.1e-5
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')
        
    x = ZeroPadding2D((3,3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64,(7,7), strides = (2,2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), name='pool1')(x)
    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')
    
    x = conv_block(x, 3,[128,128,512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3,[128,128,512], stage=3, block='b'+str(i))
    
    x = conv_block(x, 3,[256,256,1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3,[256,256,1024], stage=4, block='b'+str(i))
    
    x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c')
    
    flatten_layer = Flatten()  # instantiate the layer
    #x = flatten_layer(x)       # call it on the given tensor
    
    x_fc = AveragePooling2D((7,7), name='avg_pool')(x)
    x_fc = flatten_layer(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    
    model = Model(img_input, x_fc)
    
    if K.image_dim_ordering() =='th':
        weights_path = 'model/resnet152_weights_th.h5'
    else:
        weights_path = 'model/resnet152_weights_tf.h5'
        
    model.load_weights(weights_path, by_name=True)
    
    x_newfc = AveragePooling2D((7,7), name='avg_pool')(x)
    x_newfc = flatten_layer(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)
    
    model = Model(img_input, x_newfc)
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
