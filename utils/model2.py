import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
act = 'relu'
# act = keras.layers.LeakyReLU(alpha=0.3)
def get_model(width=128, height=128, depth=64, class_num=3):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    cx1 = layers.Conv3D(filters=64, kernel_size=3)(inputs)
    ax1 = layers.Activation(act)(cx1)
#     cx1 = layers.MaxPool3D(pool_size=2)(cx1)
    bx1 = layers.BatchNormalization()(cx1)
    
    cx2 = layers.Conv3D(filters=64, kernel_size=3)(bx1)
    ax2 = layers.Activation(act)(cx2)
#     cx2 = layers.MaxPool3D(pool_size=2)(cx2)
    bx2 = layers.BatchNormalization()(cx2)
    
    cx3 = layers.Conv3D(filters=64, kernel_size=3)(bx2)
    ax3 = layers.Activation(act)(cx3)
    px1 = layers.MaxPool3D(pool_size=2)(ax3)
    bx3 = layers.BatchNormalization()(px1)
    
    cx4 = layers.Conv3D(filters=128, kernel_size=3)(bx3)
    ax4 = layers.Activation(act)(cx4)
    px2 = layers.MaxPool3D(pool_size=2)(ax4)
    bx4 = layers.BatchNormalization()(px2)
    
    cx5 = layers.Conv3D(filters=256, kernel_size=3)(bx4)
    ax3 = layers.Activation(act)(cx5)
    px3 = layers.MaxPool3D(pool_size=2)(ax3)
    bx5 = layers.BatchNormalization()(px3)
    
    outputs2 = layers.UpSampling3D(size=2, name='upsample_1')(cx4)
    
    x = layers.GlobalAveragePooling3D()(bx4)

    x = layers.Dense(units=512)(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=256)(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(units=class_num, activation="softmax", name='classification')(x)

    # Define the model.
#     model = keras.Model(inputs, [outputs, outputs2], name="3dcnn")
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

