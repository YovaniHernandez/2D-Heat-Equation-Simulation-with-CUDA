import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio

files = sorted(glob.glob("temp.csv"))
images = []
print(files)
for f in files:
    T = np.loadtxt(f, delimiter=",")
    plt.imshow(T, cmap="hot", origin="lower", vmin=0, vmax=100)
    plt.colorbar()
    plt.title(f)
    plt.axis("off")

    plt.savefig("frame.png")
    images.append(imageio.v2.imread("frame.png"))
    plt.clf()

imageio.mimsave("heat_diffusion.gif", images, fps=30)
