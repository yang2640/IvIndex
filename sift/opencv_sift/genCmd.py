import os

root = "/home/yzhou/IvIndex/data/Images" 
filenames = os.listdir(root)
filenames.sort()
for filename in filenames:
    if filename.startswith("ukbench") and filename.endswith(".jpg"):
        imgPath = os.path.join(root, filename)
        cmd = "./sift %s" % (imgPath)
        print cmd
