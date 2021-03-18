
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#data = datasets.FashionMNIST(root="Project 2 - Active learning", download=True)
#print(data)

import idx2numpy
import numpy as np
file = '/Users/yeganehghamari/Active_ML_and_Agency1/Project 2 - Active learning/data/train-images-idx3-ubyte'
arr = idx2numpy.convert_from_file(file)

#cv.imshow("Image", arr[4], )
plt.imshow(arr[4], cmap='gray')