import pickle
import argparse
import glob
import cv2
from sklearn.neighbors import BallTree


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--VLADdescriptorPath", required = True,
	help = "Path to the file that contains the VLAD descriptors")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())

path = args["VLADdescriptorPath"]
output=args["output"]

print("indexing VLAD descriptors from " +path+ " with a ball tree:")

with open(path, 'rb') as f:
    VLAD_DS=pickle.load(f)
imageID=VLAD_DS[0]
V=VLAD_DS[1]
pathImageData=VLAD_DS[2]

tree = BallTree(V, leaf_size=512)
file=output+".pickle"
print("Dumping...")
with open(file, 'wb') as f:
	pickle.dump([imageID,tree,pathImageData], f)
print("Thank God writen to file \m/")
print("The ball tree index is saved at "+file)

