import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import radialProfile
from scipy.interpolate import griddata
import dlib
import argparse

parser = argparse.ArgumentParser(description='Process some paths.')

parser.add_argument('--m', type=int, default=2, help='N = m * n')

parser.add_argument('--n', type=int, default=2, help='N = m * n')

parser.add_argument('--detector_path', type=str, default='/media/ForensicsForest-main/shape_predictor_68_face_landmarks.dat',)

parser.add_argument('--read_path', type=str, default='/media/ForensicsForest-main/StyleGAN/train/0/', help='The path of input images')

parser.add_argument('--save_patch_path', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/patch/train/0/', help='The path of image patchs')

parser.add_argument('--save_feature_path1', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/train/0/hist/', help='The path of appearance features')

parser.add_argument('--save_feature_path2', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/train/0/spec/', help='The path of frequency features')

parser.add_argument('--save_feature_path3', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/train/0/landmarks/', help='The path of biology features')

parser.add_argument('--save_feature_path', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/train/0/', help='The path of final input features')

args = parser.parse_args()

N = 88
epsilon = 1e-8


def divide_method(img, m, n):
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (n - 1) + 0.5)

    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    img_re = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int_)
    gy = gy.astype(np.int_)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3], np.uint8)

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image


def save_patchs(title, divide_image):
    if not os.path.isdir(args.save_patch_path):
        os.makedirs(args.save_patch_path)
    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.imshow(divide_image[i, j, :])
            plt.axis('off')
            plotPath = str(title) + "+" + str(i) + str(j) + '.png'
            plt.imsave(args.save_patch_path + plotPath, divide_image[i, j, :])


def extract_histgram(save_patch_path, save_feature_path1):
    if not os.path.isdir(save_feature_path1):
        os.makedirs(save_feature_path1)

    a1 = []
    a2 = []
    a3 = []
    i = 1
    for item in os.listdir(path=save_patch_path):
        item1 = os.path.splitext(item)[0]
        index = item1.find("+")
        item1 = item1[:index]

        image = cv2.imread(os.path.join(save_patch_path, item))
        B, G, R = cv2.split(image)

        hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])
        a1.append(hist_R.flatten())

        hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
        a2.append(hist_G.flatten())

        hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
        a3.append(hist_B.flatten())

        if i % (args.m * args.n) == 0:
            a1 = np.array(a1, dtype=object)
            a2 = np.array(a2, dtype=object)
            a3 = np.array(a3, dtype=object)
            a = np.hstack((a1, a2))
            a = np.hstack((a, a3))

            np.save(save_feature_path1 + item1 + '.npy', a)
            a1 = []
            a2 = []
            a3 = []

        i = i + 1


def extract_spectrum(save_patch_path, save_feature_path2):
    if not os.path.isdir(save_feature_path2):
        os.makedirs(save_feature_path2)
    b1 = []
    b2 = []
    b3 = []
    i = 1
    for item in os.listdir(path=save_patch_path):
        item1 = os.path.splitext(item)[0]
        index = item1.find("+")
        item1 = item1[:index]

        image = cv2.imread(os.path.join(save_patch_path, item))
        B, G, R = cv2.split(image)

        f = np.fft.fft2(R)
        fshift = np.fft.fftshift(f)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
        points = np.linspace(0, N, num=psd1D.size)
        xi = np.linspace(0, N, num=N)
        interpolated = griddata(points, psd1D, xi, method='cubic')
        interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
        psd1D_org = interpolated
        psd1D_org = [float(format(x, '.2g')) for x in psd1D_org]
        b1.append(list(psd1D_org))

        f = np.fft.fft2(G)
        fshift = np.fft.fftshift(f)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
        points = np.linspace(0, N, num=psd1D.size)
        xi = np.linspace(0, N, num=N)
        interpolated = griddata(points, psd1D, xi, method='cubic')
        interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
        psd1D_org = interpolated
        psd1D_org = [float(format(x, '.2g')) for x in psd1D_org]
        b2.append(list(psd1D_org))

        f = np.fft.fft2(B)
        fshift = np.fft.fftshift(f)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
        points = np.linspace(0, N, num=psd1D.size)
        xi = np.linspace(0, N, num=N)
        interpolated = griddata(points, psd1D, xi, method='cubic')
        interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
        psd1D_org = interpolated
        psd1D_org = [float(format(x, '.2g')) for x in psd1D_org]
        b3.append(list(psd1D_org))

        if i % (args.m * args.n) == 0:
            b1 = np.array(b1, dtype=object)
            b2 = np.array(b2, dtype=object)
            b3 = np.array(b3, dtype=object)
            b = np.hstack((b1, b2))
            b = np.hstack((b, b3))

            np.save(save_feature_path2 + item1 + '.npy', b)
            b1 = []
            b2 = []
            b3 = []

        i = i + 1


def extract_landmarks(read_path, save_feature_path3):
    if not os.path.isdir(save_feature_path3):
        os.makedirs(save_feature_path3)
    for item in os.listdir(path=read_path):
        land = []
        item1 = os.path.splitext(item)[0]
        image = cv2.imread(os.path.join(read_path, item))
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args.detector_path)
        dets = detector(image, 1)

        # If the landmarks cannot be detected, -1 is used instead of the coordinate
        if (not dets):
            land = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1]
        for i, d in enumerate(dets):
            shape = predictor(image, d)
            for index, pt in enumerate(shape.parts()):
                land.append(pt.x)
                land.append(pt.y)

        land = np.array(land)
        np.save(save_feature_path3 + item1 + '.npy', land)


def merge(save_feature_path1, save_feature_path2, save_feature_path3, save_path):  # for N = 4
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    hist_1 = []
    hist_2 = []
    hist_3 = []
    hist_4 = []
    for item in os.listdir(path=save_feature_path1):
        a = np.load(save_feature_path1 + item, allow_pickle=True)
        hist_1.append(list(a[0]))
        hist_2.append(list(a[1]))
        hist_3.append(list(a[2]))
        hist_4.append(list(a[3]))
    hist_1 = np.array(hist_1, dtype=object)
    hist_2 = np.array(hist_2, dtype=object)
    hist_3 = np.array(hist_3, dtype=object)
    hist_4 = np.array(hist_4, dtype=object)

    spec_1 = []
    spec_2 = []
    spec_3 = []
    spec_4 = []
    for item in os.listdir(path=save_feature_path2):
        b = np.load(save_feature_path2 + item, allow_pickle=True)
        spec_1.append(list(b[0]))
        spec_2.append(list(b[1]))
        spec_3.append(list(b[2]))
        spec_4.append(list(b[3]))
    spec_1 = np.array(spec_1, dtype=object)
    spec_2 = np.array(spec_2, dtype=object)
    spec_3 = np.array(spec_3, dtype=object)
    spec_4 = np.array(spec_4, dtype=object)

    land_1 = []
    land_2 = []
    land_3 = []
    land_4 = []
    for item in os.listdir(path=save_feature_path3):
        land = np.load(save_feature_path3 + item)
        if land.shape != (136,):
            land = land[:136]
        land = np.expand_dims(land, 0).repeat(4, axis=0)
        land_1.append(list(land[0]))
        land_2.append(list(land[1]))
        land_3.append(list(land[2]))
        land_4.append(list(land[3]))
    land_1 = np.array(land_1, dtype=object)
    land_2 = np.array(land_2, dtype=object)
    land_3 = np.array(land_3, dtype=object)
    land_4 = np.array(land_4, dtype=object)

    X1 = np.hstack((hist_1, spec_1))
    X2 = np.hstack((hist_2, spec_2))
    X3 = np.hstack((hist_3, spec_3))
    X4 = np.hstack((hist_4, spec_4))

    X1 = np.hstack((X1, land_1))
    X2 = np.hstack((X2, land_2))
    X3 = np.hstack((X3, land_3))
    X4 = np.hstack((X4, land_4))

    print(X1.shape)
    print(X2.shape)
    print(X3.shape)
    print(X4.shape)
    np.save(save_path + 'X1.npy', X1)
    np.save(save_path + 'X2.npy', X2)
    np.save(save_path + 'X3.npy', X3)
    np.save(save_path + 'X4.npy', X4)


if __name__ == '__main__':
    for item in os.listdir(path=args.read_path):
        img = cv2.imread(os.path.join(args.read_path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        title = os.path.splitext(item)[0]

        h, w = img.shape[0], img.shape[1]
        divide_image = divide_method(img, args.m + 1, args.n + 1)
        save_patchs(title, divide_image)

    extract_histgram(args.save_patch_path, args.save_feature_path1)

    extract_spectrum(args.save_patch_path, args.save_feature_path2)

    extract_landmarks(args.read_path, args.save_feature_path3)

    merge(args.save_feature_path1, args.save_feature_path2, args.save_feature_path3, args.save_feature_path)
