import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.config.loader import Config


# Configure the prompt so that I know I am in a nested (embedded) shell
cfg = Config()
prompt_config = cfg.PromptManager
prompt_config.in_template = 'N.In <\\#>: '
prompt_config.in2_template = '   .\\D.: '
prompt_config.out_template = 'N.Out<\\#>: '

# Messages displayed when I drop into and exit the shell.
banner_msg = ("\n**Nested Interpreter:\n"
"Hit Ctrl-D to exit interpreter and continue program.\n"
"Note that if you use %kill_embedded, you can fully deactivate\n"
"This embedded instance so it will never turn on again")   
exit_msg = '**Leaving Nested interpreter'


def display_image(dataset, labels, index):
	'''Display a particular digit to screen'''
	print "Image label: ", labels[index]
	imgplot = plt.imshow(dataset[index])
	plt.show()


def preprocess_data(dataset, labels, binary_class = True):
	''' Preprocessing code
	0.  Normalize the pixel intensities to zero mean, unit variance
	1.  Extract 0 and 1 digits only from Test/Training
	2.  Append '1' feature to dataset for intercept term'''
	X_list = []
	Y_list = []

	for i in range(0, len(dataset)):
		mean = dataset[i].mean()
		std  = dataset[i].std()
		x    = (dataset[i].flatten() - mean)/std

		if binary_class == True:
			if labels[i] == 0 or labels[i] == 1:
				X_list.append(np.append(1.0, x))	
				Y_list.append(labels[i])
			
		elif binary_class == False:
			X_list.append(np.append(1.0, x))
			Y_list.append(labels[i])

	X = np.asarray(X_list)
	Y = np.asarray(Y_list)

	return X, Y

def print_performance(pred, actual):
	'''Simply takes in predicted and actual and prints performance'''
	num_incorrect = 0.0
	incorrect     = []

	for i in range(0, len(actual)):
		if pred[i] != actual[i]:
			num_incorrect += 1
			incorrect.append(i)
	print 'Incorrect indices: ', incorrect
	print ' Performance: ', float((len(actual)-num_incorrect)/len(actual))
	print

