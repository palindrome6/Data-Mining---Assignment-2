from openml.apiconnector import APIConnector
import pandas as pd
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def plot_histogram():
	class_histogram = iris['class_type']
	print class_histogram.value_counts()
	alphab = [1,2,3,0]
	frequencies = [81, 61, 4, 2]
	pos = np.arange(len(alphab))
	width = 1.0     # gives histogram aspect to the bar diagram
	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(alphab)
	plt.bar(pos, frequencies, width, color='#009999')
	plt.show()


if __name__ == "__main__":
	home_dir = os.path.expanduser("~")
	openml_dir = os.path.join(home_dir, ".openml")
	cache_dir = os.path.join(openml_dir, "cache")
	with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
		key = fh.readline().rstrip('\n')
	openml = APIConnector(cache_directory=cache_dir, apikey=key)
	dataset = openml.download_dataset(10)
	X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
	iris = pd.DataFrame(X, columns=attribute_names)
	iris['class_type'] = y



	# A
	plot_histogram()
	# B
	print len(iris)
	data_binary = iris[iris.class_type != 3]
	data_binary = data_binary[data_binary.class_type != 0]
	print len(data_binary)


	
	n_sample = data_binary['class_type']











