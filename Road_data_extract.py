
# coding: utf-8

# <h4> Code to extract and explore the Road dataset

# In[1]:


import shutil
import zipfile
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


zip_data = 'data_road.zip'
unzipped = 'data_road'

if not os.path.exists(os.getcwd() + unzipped):
    ZIP_REF = zipfile.ZipFile(os.path.join(os.getcwd(), zip_data))
    ZIP_REF.extractall()
    ZIP_REF.close()


# In[5]:


data_dir = os.getcwd()
print("Data is located in directory ----- {}".format(data_dir))
data_paths = [os.path.join(data_dir, 'data_road/training/gt_image_2'),
             os.path.join(data_dir, 'data_road/training/image_2')]


# In[10]:


random_int = np.random.randint(10, 97)
IMG_SAMPLE = os.path.join(data_paths[1], 'um_0000{}.png'.format(random_int))
plt.imshow(mpimg.imread(IMG_SAMPLE))


# In[11]:


IMG_SAMPLE = os.path.join(data_paths[0], 'um_road_0000{}.png'.format(random_int))
plt.imshow(mpimg.imread(IMG_SAMPLE))


# <h3> Let's look at the RGB content of some of the blob output images that we use for training the model

# In[50]:


for num in range(5):
    random_int = np.random.randint(10,35)
    IMG_SAMPLE = os.path.join(data_paths[0], 'um_road_0000{}.png'.format(random_int))
    print("IMAGE VECTORS = {}".format(mpimg.imread(IMG_SAMPLE)[:1]))
    print("IMAGE SHAPE = {}".format(mpimg.imread(IMG_SAMPLE).shape))


# <h4> The images are (375, 1242, 3) but primarily in the red channel of the RGB if its a road and not purely in the red channel if it is a road. The output we need consists of training a model to predict whether there is a road or not. So the output vector 
# must be of shape (Xdim, Ydim, 2) where Xdim and Ydim are the reduced dimensions of the pixel and the 2 is a boolean channel which is True if the output is red, false otherwise. In the main file, there is a function which converts these output training images into the right format.
