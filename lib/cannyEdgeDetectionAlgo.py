import numpy as np
from imageio.v3 import imread, imwrite
from io import BytesIO

import numpy as np

def convolve(image, kernel, mode='same'):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape


    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

   
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output_height = image_height
    output_width = image_width 

    output = np.zeros((output_height, output_width), dtype=image.dtype) # Use input dtype

    for y in range(output_height):
        for x in range(output_width):
            image_region = padded_image[y:y + kernel_height, x:x + kernel_width]

            output[y, x] = np.sum(image_region * kernel)

    return output


def to_grayscale(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def sobel_filters(img):
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.uint8)
    angle = np.rad2deg(theta) % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = r = 255
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            if G[i,j] >= q and G[i,j] >= r:
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    return Z

def threshold(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

def apply_canny_edge_detection(image_bytes):
    # m*m*3
    image = imread(BytesIO(image_bytes))   

    # m*m
    gray = to_grayscale(image)

    blur = convolve(gray, gaussian_kernel(5, sigma=1.0))
    G, theta = sobel_filters(blur)
    nonmax = non_max_suppression(G, theta)
    thresh, weak, strong = threshold(nonmax)
    edges = hysteresis(thresh, weak, strong)

    output_buffer = BytesIO()
    imwrite(output_buffer, edges.astype(np.uint8), format="png")
    return output_buffer.getvalue()
