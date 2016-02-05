import os

root = "/home/yzhou/HeIvIndex/data/Images" 
filenames = os.listdir(root)
filenames.sort()
for filename in filenames:
    if filename.startswith("ukbench") and filename.endswith(".jpg"):
        imgPath = os.path.join(root, filename)
        outPath = os.path.join(root, filename[:filename.rfind(".")] + ".dog.sift")
        cmd = "./sift %s --descriptors=%s" % (imgPath, outPath)
        print cmd
