import os, sys, math
from sklearn import datasets, svm
from sklearn.externals import joblib

DIGITS_PKL = "digit-clf.pkl"

def train_digits():
    digits = datasets.load_digits()
    data_train = digits.data
    label_train = digits.target
    clf = svm.SVC(gamma=0.001)
    clf.fit(data_train, label_train)
    joblib.dump(clf, DIGITS_PKL)
    print("予測モデルを保存しました=", DIGITS_PKL)
    return clf

def predict_digits(data):
    if not os.path.exists(DIGITS_PKL):
        clf = train_digits()
    clf = joblib.load(DIGITS_PKL)
    n = clf.predict([data])
    print("判定結果=", n)

def image_to_data(imagefile):
    import numpy as np
    from PIL import Image
    image = Image.open(imagefile).convert('L')
    image = image.resize((8, 8), Image.ANTIALIAS)
    img = np.asarray(image, dtype=float)
    img = np.floor(16 - 16 * (img / 256))
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.gray()
    plt.show()
    img = img.flatten()
    print(img)
    return img

def main():
    if len(sys.argv) <= 1:
        print("USAGE")
        print("python3 predict_digit.py imagefile")
        return img
    imagefile = sys.argv[1]
    data = image_to_data(imagefile)
    predict_digits(data)

if __name__ == '__main__':
    main()
