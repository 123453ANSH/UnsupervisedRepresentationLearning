import numpy as np
from PIL import Image
import glob
import torch
import os
from torch.utils.data.dataset import Dataset

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''
def getimage(array):
	array = array.numpy()
	im = np.moveaxis(array, [0,1,2], [2,0,1])
	return Image.fromarray(im)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class RotDataset(Dataset):
	def __init__(self, data_dir):
		images = None
		for file in os.listdir(data_dir):
			dict = unpickle(data_dir + '/' + file)
			if images is None:
				images = dict[b'data']
			else:
				images = np.concatenate((images, dict[b'data']), axis = 0)

		self.images = images.reshape(-1, 3, 32, 32)/255

	def __getitem__(self, index):
		"""
		Returns a torch tensor of size 3x32x32 and an int with the rotation label
		Labels:
		0 = 0 degrees
		1 = 90 degrees....
		"""
		rotation = index % 4
		image = self.images[index//4]
		image = np.rot90(image, rotation, axes = (-2, -1))
		image = torch.from_numpy(image).float()
		return image, rotation


	def __len__(self):
		return 4*len(self.images)


class Data(Dataset):
	def __init__(self, data_dir):
		#gets the data from the directory
		self.image_list = glob.glob(data_dir+'*')
		#calculates the length of image_list
		self.data_len = len(self.image_list)

	def __getitem__(self, index):
		# Get image name from the pandas df
		single_image_path = self.image_list[index]
    	# Open image
		image = Image.open(single_image_path)

    	# Do some operations on image
    	# Convert to numpy, dim = 28x28
		image_np = np.asarray(image)/255
    	# Add channel dimension, dim = 1x28x28
    	# Note: You do not need to do this if you are reading RGB images
    	# or i there is already channel dimension

		image_np = np.expand_dims(image_np, 0)
		'''
		#TODO: Convert your numpy to a tensor and get the labels
		'''
		image_tensor = torch.from_numpy(image_np).float()
		class_indicator_location = single_image_path.rfind('_c')
		label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
		return (im_as_ten, label)

	def __len__(self):
		return self.data_len
