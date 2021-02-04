from keras import backend as K
from termcolor import colored
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras
import tensorflow as tf
def heatmap_setup(weight_dir, input_layer, last_conv, data, ground_truth,model,data_set):
    model.load_weights(weight_dir)
    img_data = data
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=-1)
    data = np.transpose(img_data)

    preds = model.predict(img_data)
    pred_class = np.argmax(preds)
    pre_class_list1 = ['Embolic','Large Vessel','Lacune']
    pre_class_list2 = ['Anterior','Posterior']
    data_type = [pre_class_list1, pre_class_list2]
    pre_class_list = data_type[data_set]
    print(f'Ground Truth: {ground_truth}')
        
    print(colored(f'Prediction class: {pre_class_list[pred_class]}',"red"))

    lacune_class = model.output[:, 1]

    img_tensor = img_data
    # input layer
    input_layer = model.get_layer(input_layer)
    # last conv layer
    conv_layer = model.get_layer(last_conv)
    heatmap_model = keras.Model([model.inputs],[conv_layer.output,model.output])

    with tf.GradientTape() as gtape:
        conv_output,predictions = heatmap_model(img_tensor)
        loss = predictions[:,np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.min(heatmap)
    if max_heat ==0:
        max_heat = 1e-10
    heatmap/=max_heat
    heatmap = heatmap
    # (38, 192, 192, 28)

    data = np.squeeze(heatmap, axis=0)
    data = np.transpose(data)
    plt.figure(figsize=(12,12))
    for i in range(data.shape[0]):
        plt.subplot(1,data.shape[0],i+1)
        plt.imshow(data[i], cmap="gray")

    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0, right=1, bottom=0, top=1)
    plt.show()

