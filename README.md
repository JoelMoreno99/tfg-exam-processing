# Handwritten Examination Processing System
## Introduction
The source code of the system and the test dataset have been developed/generated under the concept of the final degree project named "Platform for assistance in exam correction: Handwritten text recognition".
For more information, please contact the author of the project: [Joel Moreno Barradas](https://www.linkedin.com/in/joel-moreno-barradas/)
## About the project
This project has developed the handwritten text recognition module of the platform for assistance in exam correction, a project from the UAB Computer Science Department that provides different management and evaluation tools. Given a PDF file with all the scanned exams of a group of students, the handwritten exam processing system implemented is able to correctly group the images corresponding to the resolution of a student and correctly recognise the student's identifying number with a high reliability and a good response time as long as the identifier is on a horizontal line on the first page of the exam. To achieve these results, images from the MNIST database have been used to train a modified LeNet-5 neural network and a fine-tuning has been performed with images of European digits. In addition, a varied test set has been generated containing different test models filled in manually with different handwriting and pen colours. This test set has been scanned using different configurations and is made available to the community together with the source code of the system.
## How to run
Running the application is as easy as downloading the repository, inserting the PDF document in the folder "tfg-exam-processing/auxiliary/documents" (e.g. "*tfg-exam-processing/auxiliary/documents/filename.pdf*") and executing the following command:
```powershell
python detectNIU.py filename.pdf
```

## Dependencies
This system is designed to work with Python 3.8.5 and requires the installation of the following module versions:
- [pdf2image (1.14.0)](https://pypi.org/project/pdf2image/) - A python module that wraps pdftoppm and pdftocairo to convert PDF to a PIL Image object.
- [ImageHash (4.2.0)](https://pypi.org/project/ImageHash/) - An image hashing library written in Python.
- [TensorFlow (2.4.1)](https://www.tensorflow.org/) - An end-to-end open source machine learning platform.
- [NumPy (1.20.1)](https://pypi.org/project/numpy/) - The fundamental package for array computing with Python.
- [sklearn (0.24.1)](https://scikit-learn.org/stable/) - A Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.
- [imutils (0.5.4)](https://pypi.org/project/imutils/) - A series of convenience functions to make basic image processing functions.
- [OpenCV (4.5.1.48)](https://pypi.org/project/opencv-python/) - A wrapper package for OpenCV Python bindings.
- [Pillow (8.1.2)](https://pypi.org/project/Pillow/) - A friendly PIL fork, the Python Imaging Library.