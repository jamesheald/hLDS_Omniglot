import os
import random as rdm
from jax import numpy as np
import numpy as onp
from matplotlib import pyplot as plt

# images vs plots
# in images it is row (y axis) then column (x axis), in plots it is x axis then y axis
# in images y = 0 (row 0) is at the top, in plots y = 0 is at the bottom
# hence in images: bottom left (0,1), bottom right (1,1), top left (0,0), top right (1,0)

def num2str(idx):

	if idx < 10:

		return '0' + str(idx)

	return str(idx)

def load_img(fn):

	I = plt.imread(fn)
	I = onp.array(I, dtype=bool)

	return I

def space_motor_to_img(pt):

	pt[1] = -pt[1]

	return pt

# Load stroke data for a character from text file
def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline !='START':
			arr = onp.fromstring(myline, dtype = float, sep = ',')
			break
	return space_motor_to_img(arr[:2]) # first two columns are coordinates, the last column is the timing data (in milliseconds)

def create_data_split(args):

	rdm.seed(args.data_seed)

	img_dir = '../../omniglot/python/images_background_small1'
	stroke_dir = '../../omniglot/python/strokes_background_small1'
	n_char = 4 # number of renditions for each character
	n_alpha = 1 # number of alphabets to show

	alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.'] # get folder names
	alphabet_names = rdm.sample(alphabet_names, n_alpha) # choose random alphabets

	train_dataset = onp.zeros((n_char, 105, 105, 2))
	for a in range(n_alpha): # for each alphabet
		
		alpha_name = alphabet_names[a]

		for c in range(n_char):

			# choose a random character from the alphabet
			character_id = rdm.randint(0, len(os.listdir(os.path.join(img_dir, alpha_name))) - 1)

			# get image and stroke directories for this character
			img_char_dir = os.path.join(img_dir, alpha_name, 'character' + num2str(character_id))
			stroke_char_dir = os.path.join(stroke_dir, alpha_name, 'character' + num2str(character_id))

			# get base file name for this character
			fn_example = os.listdir(img_char_dir)[0]
			fn_base = fn_example[:fn_example.find('_')] 

			image_instance = 1
			fn_stk = stroke_char_dir + '/' + fn_base + '_' + num2str(image_instance) + '.txt'
			fn_img = img_char_dir + '/' + fn_base + '_' + num2str(image_instance) + '.png'
			
			I = load_img(fn_img) == False # False ensures drawn pixels are 1
			motor = load_motor(fn_stk)
			
			train_dataset[c,:,:,0] = I
			train_dataset[c,int(motor[1]),int(motor[0]),1] = 1

	train_dataset = np.array(train_dataset)
	validate_dataset = train_dataset[:]

	return train_dataset, validate_dataset