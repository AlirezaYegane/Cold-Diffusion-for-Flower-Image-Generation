from pathlib import Path
from scipy.io import loadmat

root = Path("data/raw")

labels_mat = loadmat(root / "imagelabels.mat")
setid_mat = loadmat(root / "setid.mat")

labels = labels_mat["labels"].squeeze()
trn = setid_mat["trnid"].squeeze()
val = setid_mat["valid"].squeeze()
tst = setid_mat["tstid"].squeeze()

print("Total labels:", len(labels))
print("Train size:", len(trn))
print("Val size:", len(val))
print("Test size:", len(tst))
print("Min label:", labels.min())
print("Max label:", labels.max())
print("First 10 train ids:", trn[:10])
