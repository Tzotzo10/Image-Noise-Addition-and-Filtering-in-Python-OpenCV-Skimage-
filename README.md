# Image Noise Simulation and Denoising Evaluation (Python – OpenCV & Skimage)

This project demonstrates how different noise types affect image quality and how averaging filters can be used to reduce them. A grayscale image is distorted using **Salt-and-Pepper** and **Poisson noise**, then several **mean filters** are tested to restore the image. Image quality before and after denoising is evaluated using **Mean Squared Error (MSE)** and **Structural Similarity Index (SSIM).**

---

## 🔧 Features
- Convert image to grayscale
- Add Salt-and-Pepper noise
- Add Poisson noise
- Apply mean filters of sizes:
  - 3×3
  - 5×5
  - 7×7
- Evaluate results using:
  - **MSE (Mean Squared Error)**
  - **SSIM (Structural Similarity Index)**

---

## 🖼 Libraries Used
- `opencv-python`
- `numpy`
- `matplotlib`
- `random`
- `skimage.metrics`

Install all dependencies with:
```bash
pip install opencv-python numpy matplotlib scikit-image

▶️ How to Run

Place an image named jinx.jpg in the same folder as the script.

Run the Python file:

python p2.py


The program will:

Display original and noisy images

Show filtered results

Print SSIM and MSE scores

Save all generated images

📌 Output Saved Files
File	Description
gray_jinx.jpg	Grayscale image
sp_jinx.jpg	Salt-and-Pepper noisy image
poisson_jinx.jpg	Poisson noisy image
noiseless_jinx3*.jpg, noiseless_jinx5*.jpg, noiseless_jinx7*.jpg	Filtered images using 3×3, 5×5, 7×7 kernels
📈 Evaluation Metrics
Metric	Used For	Meaning
MSE	Error measurement	Lower is better
SSIM	Structural similarity	Higher is better

These metrics help compare how well each filter restores the noisy images.

✨ Future Improvements (Optional Ideas)

Compare with median filter and Gaussian filter

Support multiple input images

Plot MSE/SSIM for all filters in a bar chart

Save metrics in a CSV file

👩‍💻 Author

Papapanagiotou Georgia