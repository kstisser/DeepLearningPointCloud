# DeepLearningPointCloud

There are currently 2 pipelines:

To run the training program, run: 
python mainTrainingPipeline.py

To run the data augmentation pipeline (once it's finished), run:
python mainDataAugmentation.py

Expectations:
Point Cloud Data folder is extracted in the 'Cloud Sample Data' Directory

The following dependencies are installed on your system:
(Python3)
- matplotlib
- numpy
- tensorflow 2/keras
- open3d
- pandas

Recommendation:
Install Anaconda, create an environment and install jupyter, tensorflow, and if necessary numpy and pandas
In a terminal for that environment run:
pip install open3d

This has been tested on a Windows 10 system.

Current Code Structure:
![Architecture](https://github.com/kstisser/DeepLearningPointCloud/blob/main/Documentation/OriginalPlanPointCloudFaceFindingArchitecture.jpg)
