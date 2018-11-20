{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> Following the Udacity nanodegree course, I implement an Encoder Decoder network using VGG16 and transfer learning. To train and predict, I use the KITTI road dataset to detect roads and lanes. \n",
    "\n",
    "<h3> The encoding portion of this network is done using a pre-trained VGG model. To this we add a FCN (fully convolutional network) as the decoder layers. I introduce skip connections and upsample the downsampled VGG images. \n",
    "\n",
    "<h4> We will try to run this model on Google's Collab on a GPU for training. The following code also references a separate helper function to download and load the pretrained VGG model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\stenatu\\\\deep_learning_models'"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.getcwd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import os.path\n",
    "import warnings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.11.0\n",
      "Please use tensorflow version 1.0 and above\n"
     ]
    }
   ],
   "source": [
    "# check version\n",
    "print(tf.__version__)\n",
    "print(\"Please use tensorflow version 1.0 and above\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Please train model on GPU\n"
     ]
    }
   ],
   "source": [
    "# check for GPU -- in Google Collab this will use the GPU\n",
    "if not tf.test.gpu_device_name():\n",
    "        print(\"Please train model on GPU\")\n",
    "else:\n",
    "    print('GPU Device {}'.format(tf.test.gpu_device_name))\n",
    "    device_name = tf.test.gpu_device_name()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_vgg(sess, vgg_path):\n",
    "    ''' Given a path to the VGG model, load the model. If the model is not available, download the model.\n",
    "    : return: tuple of tensors containing the outputs from VGG layers 3, 4, and 7 as well as the input image '''\n",
    "    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)\n",
    "    \n",
    "    graph = tf.get_default_graph()\n",
    "    image_input = graph.get_tensor_by_name('image_input:0')\n",
    "    keep_prob = graph.get_tensor_by_name('keep_prob:0')\n",
    "    layer_3 = graph.get_tensor_by_name('layer3_out:0')\n",
    "    layer_4 = graph.get_tensor_by_name('layer4_out:0')\n",
    "    layer_7 = graph.get_tensor_by_name('layer7_out:0')\n",
    "    \n",
    "    return image_input, keep_prob, layer_3, layer_4, layer_7\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Restoring parameters from C:\\Users\\stenatu\\deep_learning_models\\vgg/vgg\\variables\\variables\n",
      "Tensor(\"image_input:0\", shape=(?, ?, ?, 3), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "# test this out\n",
    "vgg_path = os.path.join(os.getcwd(), 'vgg/vgg')\n",
    "with tf.Session() as session:\n",
    "        \n",
    "    # Returns the three layers, keep probability and input layer from the vgg architecture\n",
    "    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)\n",
    "    print(image_input)\n",
    "    session.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4> Write the FCN Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):\n",
    "    ''' builds the FCN layer on top of the layer 7 of the VGG layer'''\n",
    "    \n",
    "    #replace final fully connected layer by 1 X 1 conv layer\n",
    "    \n",
    "    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters = num_classes, kernel_size=1, name = \"fcn8\") \n",
    "    \n",
    "    #upsample to same size as output of layer 4 to add a skip connection to layer 4\n",
    "    fcn9 = tf.layers.conv2d_transpose(fcn8, filters = vgg_layer4_out.get_shape().as_list()[-1],\n",
    "                                     kernel_size = 4, strides = (2, 2), padding = 'SAME',\n",
    "                                     name = 'fcn9')\n",
    "    \n",
    "    # add a skip connection between 9 and 4\n",
    "    fcn9_skip = tf.add(fcn9, vgg_layer4_out, name = 'fc9_skip')\n",
    "    \n",
    "    #upsample again\n",
    "    fcn10 = tf.layers.conv2d_transpose(fcn9_skip, filters = vgg_layer3_out.get_shape().as_list()[-1],\n",
    "                                     kernel_size = 4, strides = (2, 2), padding = 'SAME',\n",
    "                                     name = 'fcn10')\n",
    "    \n",
    "    # add another skip connection\n",
    "    fcn10_skip = tf.add(fcn10, vgg_layer3_out, name = 'fc10_skip')\n",
    "    \n",
    "    #upsample again\n",
    "    fcn11 = tf.layers.conv2d_transpose(fcn10_skip, filters = num_classes,\n",
    "                                     kernel_size = 16, strides = (8, 8), padding = 'SAME',\n",
    "                                     name = 'fcn11')   \n",
    "    \n",
    "    \n",
    "    return fcn11\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4> Write the Optimizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "def optimizer(nn_layer, correct_label, num_classes, learning_rate):\n",
    "    '''write the optimization functions to train the weights of the neural network'''    \n",
    "    logits = tf.reshape(nn_layer, (-1, num_classes), name = 'fcn_logits')\n",
    "    correct_labels_reshaped = tf.reshape(correct_label, (-1, num_classes))\n",
    "    \n",
    "    # use cross entropy as the loss metric\n",
    "    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = correct_labels_reshaped)\n",
    "    #take mean of total loss\n",
    "    loss_op = tf.reduce_mean(cross_entropy, name = 'fcn_loss')\n",
    "    \n",
    "    #optmizer = Adam\n",
    "    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op)\n",
    "    \n",
    "    return logits, train_op, loss_op"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Write a custom train function to train the model. Later on I will change this to work on Google Cloud Platform by using train and evaluate. hparams consists of the parameters used to train the model which will be passed as a dictionary entered as a json input. For now we just enter is here. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "from glob import glob\n",
    "import random\n",
    "import re\n",
    "import numpy as np\n",
    "import skimage.transform\n",
    "import imageio\n",
    "\n",
    "def gen_batches_fn(data_path, image_shape):\n",
    "    ''' Generates a function which creates batches of training data'''\n",
    "    \n",
    "    def get_batches_fn(batch_size):\n",
    "        ''' given the path to the images, generates random batches'''\n",
    "\n",
    "        image_paths = glob(os.path.join(data_path, 'training\\image_2', '*.png'))\n",
    "        label_paths = {\n",
    "            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for \n",
    "             path in glob(os.path.join(data_path, 'training\\gt_image_2', '*_road_*.png'))}\n",
    "        \n",
    "        background_color = [255, 0, 0]\n",
    "    \n",
    "        random.shuffle(image_paths)\n",
    "        for batch_i in range(0, len(image_paths), batch_size):\n",
    "            images = []\n",
    "            gt_images = []\n",
    "            for image_file in image_paths[batch_i:batch_i+batch_size]:\n",
    "                gt_image_file = label_paths[os.path.basename(image_file)]\n",
    "                image = skimage.transform.resize(imageio.imread(image_file), image_shape)\n",
    "                gt_image = skimage.transform.resize(imageio.imread(gt_image_file), image_shape)\n",
    "\n",
    "                gt_bg = np.all(gt_image == background_color, axis=2)\n",
    "                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)\n",
    "                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)\n",
    "                images.append(image)\n",
    "                gt_images.append(gt_image)\n",
    "\n",
    "            yield np.array(images), np.array(gt_images)    \n",
    "    return get_batches_fn\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_fn(sess, get_batches_fn, train_op, cross_entropy_loss,\n",
    "            input_image, correct_label, image_shape, num_classes, num_epochs, batch_size):\n",
    "    ''' function which trains the model'''\n",
    "   # dropout = tf.placeholder(tf.float32)\n",
    "\n",
    "    for epoch in range(num_epochs):\n",
    "        total_loss = 0\n",
    "        for X_batch, gt_batch in get_batches_fn(batch_size):\n",
    "            loss, _ = sess.run([cross_entropy_loss, train_op], \n",
    "                              feed_dict = {input_image: X_batch, correct_label: gt_batch})\n",
    "            total_loss+= loss\n",
    "        print(\"EPOCH {}\".format(epoch+1))\n",
    "        print(\"LOSS = {:.3f}\".format(total_loss))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h5> Write the function which runs the code. During Model serving, user can provide hparams. This could be done in a separate notebook called task.py. This notebook then reads in hparams from task.py."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_PATH = os.path.join(os.getcwd(), 'data_road')\n",
    "vgg_path = os.path.join(os.getcwd(), 'vgg/vgg')\n",
    "\n",
    "hparams = {'IMAGE_SHAPE':(160, 576),\n",
    "    'NUM_CLASSES': 2,\n",
    "    'NUM_EPOCHS': 1,\n",
    "    'BATCH_SIZE': 16,\n",
    "    'DROPOUT': 0.75,\n",
    "    'LEARNING_RATE': 0.001}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<generator object gen_batches_fn.<locals>.get_batches_fn at 0x0000016E0F11BAF0>"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# test if the batch function is working\n",
    "get_batches_fn = gen_batches_fn(DATA_PATH, hparams.get('IMAGE_SHAPE'))\n",
    "get_batches_fn(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run():\n",
    "    tf.reset_default_graph() # ensures that names don't get reused and the graph is reset everytime run is called.\n",
    "    image_shape = hparams.get('IMAGE_SHAPE')\n",
    "    num_classes = hparams.get('NUM_CLASSES')\n",
    "    batch_size = hparams.get('BATCH_SIZE')\n",
    "    learning_rate = hparams.get('LEARNING_RATE')\n",
    "    num_epochs = hparams.get('NUM_EPOCHS')\n",
    "    \n",
    "    get_batches_fn = gen_batches_fn(DATA_PATH, image_shape)\n",
    "    correct_label = tf.placeholder(tf.float32, shape = [None, image_shape[0], image_shape[1], num_classes])\n",
    "    \n",
    "    with tf.Session() as sess:\n",
    "        \n",
    "        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)\n",
    "        \n",
    "        model_output = layers(layer3, layer4, layer7, num_classes)\n",
    "        \n",
    "        logits, train_op, cross_entropy_loss = optimizer(model_output, correct_label, num_classes, learning_rate)\n",
    "        \n",
    "        sess.run(tf.global_variables_initializer())\n",
    "        sess.run(tf.local_variables_initializer())\n",
    "        \n",
    "        print(\"Model build successful, starting training\")\n",
    "#        sess.close() \n",
    "        train_fn(sess, get_batches_fn, \n",
    "             train_op, cross_entropy_loss, image_input,\n",
    "             correct_label,image_shape, num_classes, num_epochs, batch_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h5> Main Function for Model Serving"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == 'main':\n",
    "    run()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
