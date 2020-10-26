#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:44:48 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

"""

from __future__ import print_function, division
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class SI_P300Datasets(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, dataset_name, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.d1 = None
		with open(dataset_name + '.pickle', 'rb') as fh:
			self.d1 = pickle.load(fh)
        
		self.transform = transform
		
	def import_subjects(self, sub, augment=True):
		"""      
		Input: d1 - is list consisting of subject-specific epochs in MNE structure
		Example usage:
			subjectIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            
			for sub in subjectIndex:
				xvalid = subject_specific(sub, d1)          
		"""
		pos_str = 'Target'
		neg_str = 'NonTarget'
	
		self.data, pos, neg = {}, [] , []    
		if len(sub) > 1: # multiple subjects             
			for jj in sub:                
				print('Loading subjects:', jj)   
				dat = self.d1[jj]                                     
				pos.append(dat[pos_str].get_data())
				neg.append(dat[neg_str].get_data())
		else: 
			print('Loading subject:', sub[0])  
			dat = self.d1[sub[0]]
			pos.append(dat[pos_str].get_data())
			neg.append(dat[neg_str].get_data())
		
		if augment == True:
			for ii in range(len(pos)):
				# subject specific upsampling of minority class 
				targets = pos[ii]              
				for j in range((neg[ii].shape[0]//pos[ii].shape[0]) - 1): 
					targets = np.concatenate([pos[ii], targets])                    
				pos[ii] = targets  
		
		for ii in range(len(pos)):            
			X = np.concatenate([pos[ii].astype('float32'), neg[ii].astype('float32')])
			Y = np.concatenate([np.ones(pos[ii].shape[0]).astype('float32'), 
								np.zeros(neg[ii].shape[0]).astype('float32')])       
			try:
				self.data['xtrain'] = np.concatenate((self.data['xtrain'], X))
				self.data['ytrain'] = np.concatenate((self.data['ytrain'], Y))
			except:
				self.data['xtrain'] = X
				self.data['ytrain'] = Y
		return self.data
    
	def apply_normalization(self):
		self.scaler = NDStandardScaler()
		self.scaler.fit_transform(self.data['xtrain'])
            
	def apply_normalization_to_test(self, scaler):
		self.data['xtrain'] = scaler.transform(self.data['xtrain'])

	def get_normalization_params(self):
		return self.scaler

	def __len__(self):
		return len(self.data['xtrain'])

	def __getitem__(self, idx):
		try:
			sample = self.data['xtrain'][idx]
			label = self.data['ytrain'][idx]
			if self.transform:
				sample = self.transform(sample)
			return sample, label
		except:
			print("Data is not imported yet. Please first import subjects using import_subjects method.")


class Reshape(object):
    """Rescale the image in a sample to a given size using duplicating

    Args:
        output_size (tuple or int):
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = int(self.output_size), int(self.output_size)

        img = self.pad_by_duplicating(sample, new_h, new_w)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        img = img[None, :, :] * np.ones(3, dtype=int)[:, None, None]

        return torch.from_numpy(img)
    
    def pad_by_duplicating(self, x, desired_height=224, desired_width=224): #duplicate signal until the desired new array is full
        x_height, x_width = x.shape[0], x.shape[1]
        new_x = np.zeros((desired_height, desired_width))
        for nhx in range(0, desired_height, x_height):
            for nwx in range(0, desired_width, x_width):
                new_x[nhx:min(nhx+x_height, desired_height), nwx:min(nwx+x_width, desired_width)] = x[0:min(x_height, desired_height-nhx), 0:min(x_width, desired_width-nwx)]
        return new_x
    
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X 