"""
 This file consists of a code to split the data into training, validation and test subsets.
 It is important to make sure that images from the same patient won't end up in different subsets.
 """

from pathlib import Path
import re
import json


def extract_sub(s: str):
    """function to extract the subject code from the path using reegex"""
    subject = re.search(r"sub-\d+", s)[0]
    return subject


# path to the data directory
base_path = "/Shared/sinapse/chdi_bids/PREDICTHD_BIDS/derivatives/physicalACPC/"

# extract the paths of t1w images into a array
i = 0
t1w_paths = []
for t1w_file in Path(base_path).glob("sub-*/ses-*/*T1w.nii.gz"):
    t1w_paths.append(str(t1w_file))
    i += 1

print(i)  # print the number of paths
t1w_paths.sort()  # sort the array to organize the paths in the array

# calculate the proportions of the data for the split into train, val and test
num_of_files = len(t1w_paths)
print(num_of_files)
train_num = int(num_of_files * 0.8)
val_num = int(num_of_files * 0.1)

# using the calculated numbers split the data
train_t1w = t1w_paths[:train_num]
val_t1w = t1w_paths[train_num : train_num + val_num]
test_t1w = t1w_paths[train_num + val_num :]

# those 3 loops make sure that images of one subject can be only in one dataset
# (to ensure we want test on the same subject we trained on)
for i in train_t1w:
    subject = extract_sub(i)
    for j in val_t1w:
        if subject in j:
            train_t1w.append(j)
            val_t1w.remove(j)

for i in train_t1w:
    subject = extract_sub(i)
    for j in test_t1w:
        if subject in j:
            train_t1w.append(j)
            test_t1w.remove(j)

for i in val_t1w:
    subject = extract_sub(i)
    for j in test_t1w:
        if subject in j:
            val_t1w.append(j)
            test_t1w.remove(j)

print(len(train_t1w) + len(val_t1w) + len(test_t1w))

# create a dictionary
t1w_dict = {"train": train_t1w, "val": val_t1w, "test": test_t1w}

# write out the dictionary as a json file
json_object = json.dumps(t1w_dict)
# Writing to sample.json
with open("T1w_paths.json", "w") as outfile:
    outfile.write(json_object)
