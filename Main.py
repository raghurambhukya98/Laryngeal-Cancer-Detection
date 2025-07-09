import numpy as np
import pandas as pd
import os
import cv2 as cv


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


# Read Dataset 1
an = 0
if an == 1:
    path = './1003200/laryngeal dataset/laryngeal dataset'
    in_dir = os.listdir(path)
    Image = []
    Target = []
    for i in range(len(in_dir)):
        out_dir = path + '/' + in_dir[i]
        fold = os.listdir(out_dir)
        for j in range(len(fold)):
            dir = out_dir + '/' + fold[j]
            files = os.listdir(dir)
            for k in range(len(files)):
                filename = dir + '/' + files[k]
                img = ReadImage(filename)
                Image.append(img)
                Target.append(j)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Target_1.npy', Shuffled_Target)

