import pickle
import numpy as np
from sklearn.neighbors import BallTree

path = "discogs_hot_histos.pickle"
with open(path, 'rb') as f:
    pkl = pickle.load(f)
    img_names = pkl[0]
    histos = pkl[1]
print(len(histos))
new_histos = []
for histo in histos:
    histo = np.reshape(histo, 46080)
#    print(histo.shape)
    new_histos.append(histo)

new_histos = np.asarray(new_histos)
print("indexing Histos from "+path+ " with a ball tree:")
tree = BallTree(new_histos, leaf_size=512)

file="discogs_hot_tree_histosonly.pickle"
pathofimage = "/home/comp/e4252392/discogs_hot"
with open(file, 'wb') as f:
    pickle.dump([img_names,tree,pathofimage], f)
print("The ball tree index is saved at "+file)
