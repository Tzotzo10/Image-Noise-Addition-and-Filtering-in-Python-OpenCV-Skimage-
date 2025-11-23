import cv2
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

#read original image
originalImage = cv2.imread('jinx.jpg')

#turn original image to gray scale
gray_image=cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

#defining salt and pepper function
def SaltAndPepper(originalImage, prob):
    noisy_image = np.zeros(originalImage.shape, np.uint8)
    for colInd in range(originalImage.shape[0]):
        for rowInd in range(originalImage.shape[1]):
            rand = random.random()
            if rand < prob:
                noisy_image[rowInd][colInd] = 0
            elif rand > (1 - prob):
                noisy_image[rowInd][colInd] = 255
            else:
                noisy_image[rowInd][colInd] = originalImage[rowInd][colInd]
    return noisy_image

#print the original image
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', originalImage)
cv2.waitKey()

#print the gray image
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey()

#save the gray image
plt.imsave('gray_jinx.jpg', gray_image, cmap='gray', format='jpg')

#call salt and pepper function
salt_and_pepper_img=SaltAndPepper(gray_image, 0.10)

#adding poisson noise
poisson_img = np.random.poisson(gray_image / 255.0 * 0.9) / 0.9 * 255

#print the new salt and pepper image
cv2.namedWindow('Salt and Pepper Image', cv2.WINDOW_NORMAL)
cv2.imshow('Salt and Pepper Image', salt_and_pepper_img)
cv2.waitKey()

#save the salt and pepper image
plt.imsave('sp_jinx.jpg', salt_and_pepper_img, cmap='gray', format='jpg')

#print the new poisson image
cv2.namedWindow('Poisson Image', cv2.WINDOW_NORMAL)
cv2.imshow('Poisson Image', poisson_img)
cv2.waitKey()

#save the poisson image
plt.imsave('poisson_jinx.jpg', poisson_img, cmap='gray', format='jpg')

#apply some filters
kernel3 = np.ones((3,3), np.float32)/9
kernel5 = np.ones((5,5), np.float32)/25                                              #averaging filter
kernel7 = np.ones((7,7), np.float32)/49

#apply the kernel
noiseless_img3 = cv2.filter2D(salt_and_pepper_img,-1, kernel3)
noiseless_img5 = cv2.filter2D(salt_and_pepper_img,-1, kernel5)
noiseless_img7 = cv2.filter2D(salt_and_pepper_img,-1, kernel7)
noiseless_img3p = cv2.filter2D(poisson_img,-1, kernel3)
noiseless_img5p = cv2.filter2D(poisson_img,-1, kernel5)
noiseless_img7p = cv2.filter2D(poisson_img,-1, kernel7)

#call mse function
mse3=mse(gray_image, noiseless_img3)
mse5=mse(gray_image, noiseless_img5)
mse7=mse(gray_image, noiseless_img7)
mse3p=mse(gray_image, noiseless_img3p)
mse5p=mse(gray_image, noiseless_img5p)
mse7p=mse(gray_image, noiseless_img7p)

#print the noisless images
cv2.namedWindow('Noiseless Image 3', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 3', noiseless_img3)
#save the noiseless image image 3 sp
plt.imsave('noiseless_jinx3.jpg', noiseless_img3, cmap='gray', format='jpg')
cv2.namedWindow('Noiseless Image 3p', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 3p', noiseless_img3p)
#save the noiseless image 3 poisson
plt.imsave('noiseless_jinx3p.jpg', noiseless_img3p, cmap='gray', format='jpg')
simil_score3, _ = ssim(gray_image, noiseless_img3, full=True)
print('SSIM score after salt and pepper noise is:{:.3f}'.format(simil_score3))
print('MSE difference between original image and noiseless image after salt and pepper noise is:{:.3f}'.format(mse3))
simil_score3p, _ = ssim(gray_image, noiseless_img3p, full=True)
print('SSIM score after poisson noise is:{:.3f}'.format(simil_score3p))
print('MSE difference between original image and noiseless image after poisson noise is:{:.3f}'.format(mse3p))
cv2.waitKey()
cv2.namedWindow('Noiseless Image 5', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 5', noiseless_img5)
#save the noiseless image 5 sp
plt.imsave('noiseless_jinx5.jpg', noiseless_img5, cmap='gray', format='jpg')
cv2.namedWindow('Noiseless Image 5p', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 5p', noiseless_img5p)
#save the noiseless image 5 poisson
plt.imsave('noiseless_jinx5p.jpg', noiseless_img5p, cmap='gray', format='jpg')
simil_score5, _ = ssim(gray_image, noiseless_img5, full=True)
print('SSIM score after salt and pepper noise is:{:.3f}'.format(simil_score5))
print('MSE difference between original image and noiseless image after salt and pepper noise is:{:.3f}'.format(mse5))
simil_score5p, _ = ssim(gray_image, noiseless_img5p, full=True)
print('SSIM score after poisson noise is:{:.3f}'.format(simil_score5p))
print('MSE difference between original image and noiseless image after poisson noise is:{:.3f}'.format(mse5p))
cv2.waitKey()
cv2.namedWindow('Noiseless Image 7', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 7', noiseless_img7)
#save the noiseless image 7 sp
plt.imsave('noiseless_jinx7.jpg', noiseless_img7, cmap='gray', format='jpg')
cv2.namedWindow('Noiseless Image 7p', cv2.WINDOW_NORMAL)
cv2.imshow('Noiseless Image 7p', noiseless_img7p)
#save the noiseless image 7 poisson
plt.imsave('noiseless_jinx7p.jpg', noiseless_img7p, cmap='gray', format='jpg')
simil_score7, _ = ssim(gray_image, noiseless_img7, full=True)
print('SSIM score after salt and pepper noise is:{:.3f}'.format(simil_score7))
print('MSE difference between original image and noiseless image after salt and pepper noise is:{:.3f}'.format(mse7))
simil_score7p, _ = ssim(gray_image, noiseless_img7p, full=True)
print('SSIM score after poisson noise is:{:.3f}'.format(simil_score7p))
print('MSE difference between original image and noiseless image after poisson noise is:{:.3f}'.format(mse7p))

cv2.waitKey()
cv2.destroyAllWindows()