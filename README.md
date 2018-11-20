# Semantic-Segmentation
Udacity Course -implement FCN for image segmentation

I implement a FCN following the Medium post by James Le (https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef)

The model builds a decoder layer on top of pretrained VGG8. The decoder network expands the output of the VGG8 using conv2d_transpose layers to conver the image embedding back to the original size for pixel wise comparison to the output. The consolution operator is invertible, so the deconvolution can be readily done. Skip connections are added to improve training accuracy to allow better flow of gradients between the encoder and decoder layers.

Unlike a usual convolutional neural network, where the last layer is a fully connected layer, here we use a convolutional layer which yields an image the same size as the original image but blurred becuase of the convolutional filters we apply. 
We train this on segmented version of the input images at a pixel level. 

Cross-entropy loss is used to train the model.
