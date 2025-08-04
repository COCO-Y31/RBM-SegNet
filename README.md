# Watershed boundary extraction from digital elevation models using RBM-SegNet
This repository provides the code for RBM-SegNet, a semantic segmentation model designed for accurate watershed boundaries extraction from DEMs. The model  improves boundaries extraction by addressing the limitations of conventional threshold-based methods.

# Requirements

- CUDA-enabled GPU
- Python 3.8.2
- PyTorch 1.7.2
- GDAL  3.4.1
- torchvision  0.11.2
- opencv-python 4.6.0.66

## Code Structure

- Run `python train.py` to train RBM-SegNet.
- Run `python test.py` to predict on the trained RBM-SegNet.
- Download the trained RBM-SegNet and put it in the "weights" folder to predict samples.

### Download trained RBM-SegNet
- The trained model is available for [Google Drive](https://drive.google.com/file/d/11Q0cxMNdnqO0BltIl37PKmRpXGoJfTaL/view?usp=drive_link). To ensure proper access and usage, please follow these steps:

  Click on the Google Drive link.

  Send a request for access by clicking the "Request" button.
  Once your access is granted, you can download the model file.
  Thank you for your understanding, and please feel free to reach out if you encounter any issues.