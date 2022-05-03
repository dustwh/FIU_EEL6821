import numpy as np
from scipy.ndimage.filters import generic_filter
from scipy.ndimage import imread
#now imread() won't be supported, I used conda to downgrade module scipy, but your could use imageio.imread also.

# Load image
#this is a logo of Google
with np.DataSource().open("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "rb") as f:
    img = imageio.imread(f, mode="I")
#this is a landscape image
with np.DataSource().open("https://cdn.pixabay.com/photo/2015/12/01/20/28/road-1072823_1280.jpg", "rb") as f:
    img = imageio.imread(f, mode="I")

# Apply the operator
def edge_operator(P):
    return (np.abs((P[0] + 2 * P[1] + P[2]) - (P[6] + 2 * P[7] + P[8])) +
            np.abs((P[2] + 2 * P[6] + P[7]) - (P[0] + 2 * P[3] + P[6])))
G = generic_filter(img, edge_operator, (3, 3))
