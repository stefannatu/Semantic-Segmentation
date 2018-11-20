
# coding: utf-8

# <h2> Following the Udacity nanodegree course, I implement an Encoder Decoder network using VGG16 and transfer learning. To train and predict, I use the KITTI road dataset to detect roads and lanes. 
# 
# <h3> The encoding portion of this network is done using a pre-trained VGG model. To this we add a FCN (fully convolutional network) as the decoder layers. I introduce skip connections and upsample the downsampled VGG images. 
# 
# <h4> We will try to run this model on Google's Collab on a GPU for training. The following code also references a separate helper function to download and load the pretrained VGG model.

# In[117]:


os.getcwd()


# In[3]:


import tensorflow as tf
import os.path
import warnings


# In[2]:


# check version
print(tf.__version__)
print("Please use tensorflow version 1.0 and above")


# In[108]:


# check for GPU -- in Google Collab this will use the GPU
if not tf.test.gpu_device_name():
        print("Please train model on GPU")
else:
    print('GPU Device {}'.format(tf.test.gpu_device_name))
    device_name = tf.test.gpu_device_name()


# In[6]:


def load_vgg(sess, vgg_path):
    ''' Given a path to the VGG model, load the model. If the model is not available, download the model.
    : return: tuple of tensors containing the outputs from VGG layers 3, 4, and 7 as well as the input image '''
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer_3 = graph.get_tensor_by_name('layer3_out:0')
    layer_4 = graph.get_tensor_by_name('layer4_out:0')
    layer_7 = graph.get_tensor_by_name('layer7_out:0')
    
    return image_input, keep_prob, layer_3, layer_4, layer_7
    


# In[11]:


# test this out
vgg_path = os.path.join(os.getcwd(), 'vgg/vgg')
with tf.Session() as session:
        
    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)
    print(image_input)
    session.close()


# <h4> Write the FCN Model

# In[112]:


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    ''' builds the FCN layer on top of the layer 7 of the VGG layer'''
    
    #replace final fully connected layer by 1 X 1 conv layer
    
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters = num_classes, kernel_size=1, name = "fcn8") 
    
    #upsample to same size as output of layer 4 to add a skip connection to layer 4
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters = vgg_layer4_out.get_shape().as_list()[-1],
                                     kernel_size = 4, strides = (2, 2), padding = 'SAME',
                                     name = 'fcn9')
    
    # add a skip connection between 9 and 4
    fcn9_skip = tf.add(fcn9, vgg_layer4_out, name = 'fc9_skip')
    
    #upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip, filters = vgg_layer3_out.get_shape().as_list()[-1],
                                     kernel_size = 4, strides = (2, 2), padding = 'SAME',
                                     name = 'fcn10')
    
    # add another skip connection
    fcn10_skip = tf.add(fcn10, vgg_layer3_out, name = 'fc10_skip')
    
    #upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip, filters = num_classes,
                                     kernel_size = 16, strides = (8, 8), padding = 'SAME',
                                     name = 'fcn11')   
    
    
    return fcn11
    


# <h4> Write the Optimizer

# In[84]:


def optimizer(nn_layer, correct_label, num_classes, learning_rate):
    '''write the optimization functions to train the weights of the neural network'''    
    logits = tf.reshape(nn_layer, (-1, num_classes), name = 'fcn_logits')
    correct_labels_reshaped = tf.reshape(correct_label, (-1, num_classes))
    
    # use cross entropy as the loss metric
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = correct_labels_reshaped)
    #take mean of total loss
    loss_op = tf.reduce_mean(cross_entropy, name = 'fcn_loss')
    
    #optmizer = Adam
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op)
    
    return logits, train_op, loss_op


# <h3> Write a custom train function to train the model. Later on I will change this to work on Google Cloud Platform by using train and evaluate. hparams consists of the parameters used to train the model which will be passed as a dictionary entered as a json input. For now we just enter is here. 

# In[182]:


from glob import glob
import random
import re
import numpy as np
import skimage.transform
import imageio

def gen_batches_fn(data_path, image_shape):
    ''' Generates a function which creates batches of training data'''
    
    def get_batches_fn(batch_size):
        ''' given the path to the images, generates random batches'''

        image_paths = glob(os.path.join(data_path, 'training\image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for 
             path in glob(os.path.join(data_path, 'training\gt_image_2', '*_road_*.png'))}
        
        background_color = [255, 0, 0]
    
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = skimage.transform.resize(imageio.imread(image_file), image_shape)
                gt_image = skimage.transform.resize(imageio.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)    
    return get_batches_fn


# In[123]:


def train_fn(sess, get_batches_fn, train_op, cross_entropy_loss,
            input_image, correct_label, image_shape, num_classes, num_epochs, batch_size):
    ''' function which trains the model'''
   # dropout = tf.placeholder(tf.float32)

    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], 
                              feed_dict = {input_image: X_batch, correct_label: gt_batch})
            total_loss+= loss
        print("EPOCH {}".format(epoch+1))
        print("LOSS = {:.3f}".format(total_loss))


# <h5> Write the function which runs the code. During Model serving, user can provide hparams. This could be done in a separate notebook called task.py. This notebook then reads in hparams from task.py.

# In[118]:


DATA_PATH = os.path.join(os.getcwd(), 'data_road')
vgg_path = os.path.join(os.getcwd(), 'vgg/vgg')

hparams = {'IMAGE_SHAPE':(160, 576),
    'NUM_CLASSES': 2,
    'NUM_EPOCHS': 1,
    'BATCH_SIZE': 16,
    'DROPOUT': 0.75,
    'LEARNING_RATE': 0.001}


# In[136]:


# test if the batch function is working
get_batches_fn = gen_batches_fn(DATA_PATH, hparams.get('IMAGE_SHAPE'))
get_batches_fn(5)


# In[142]:


def run():
    tf.reset_default_graph() # ensures that names don't get reused and the graph is reset everytime run is called.
    image_shape = hparams.get('IMAGE_SHAPE')
    num_classes = hparams.get('NUM_CLASSES')
    batch_size = hparams.get('BATCH_SIZE')
    learning_rate = hparams.get('LEARNING_RATE')
    num_epochs = hparams.get('NUM_EPOCHS')
    
    get_batches_fn = gen_batches_fn(DATA_PATH, image_shape)
    correct_label = tf.placeholder(tf.float32, shape = [None, image_shape[0], image_shape[1], num_classes])
    
    with tf.Session() as sess:
        
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        model_output = layers(layer3, layer4, layer7, num_classes)
        
        logits, train_op, cross_entropy_loss = optimizer(model_output, correct_label, num_classes, learning_rate)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print("Model build successful, starting training")
#        sess.close() 
        train_fn(sess, get_batches_fn, 
             train_op, cross_entropy_loss, image_input,
             correct_label,image_shape, num_classes, num_epochs, batch_size)


# <h5> Main Function for Model Serving

# In[ ]:


if __name__ == 'main':
    run()

