#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np

file_name = "p5_image.gif"

# From http://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

img_data = load_image(file_name)
for x in np.nditer(img_data, op_flags=['readwrite']):
    x[...] = 1 - x

U, s, V = np.linalg.svd(img_data, full_matrices=False)

def run_svd(k):
    compressed = np.dot(np.dot(np.array(U[:, :k]), np.diag(s[:k])), np.array(V[:k, :]))
    # compressed[compressed > 1] = 1
    # compressed[compressed < 0] = 0
    norm = colors.Normalize(compressed.min(), compressed.max())
    compressed = norm(compressed)
    for x in np.nditer(compressed, op_flags=['readwrite']):
        x[...] = 1 - x
    if (k == 300):
        plt.figure()
        plt.imshow(compressed, cmap='gray')
        plt.show()

dimensions = [1, 3, 10, 20, 50, 100, 150, 200, 300, 342]
for k in dimensions:
    run_svd(k)