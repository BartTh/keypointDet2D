import os, glob, re

name_map = {
    "Set_1": "Set_3",
}

for root, dirs, files in os.walk(r"X:\KeypointDet\images3"):
    for f in files:
        for name in name_map.keys():
            if re.search(name, f) != None:
                new_name = re.sub(name, name_map[name], f)
                try:
                    os.rename(os.path.join(root, f), os.path.join(root, new_name))
                except OSError:
                    print("No such file or directory!")
