import open3d as o3d
import numpy as np
import os
import sys
sys.path.append("....")
import defs

class PointCloud:
    def __init__(self, rootpath, pcType):
        self.pcType = pcType
        self.rootpath = rootpath
        self.facepath = os.path.join(rootpath, "face_segment.pcd")
        self.allpath = os.path.join(rootpath, "PointCloudCapture.pcd")
        self.face3d = o3d.io.read_point_cloud(self.facepath)
        self.all3d = o3d.io.read_point_cloud(self.allpath)
        self.faceseg = np.asarray(self.face3d.points)
        self.allseg = np.asarray(self.all3d.points)
        print("Read in Point Cloud object " + rootpath)
        print("Face shape: ", self.faceseg.shape)
        print("All shape: ", self.allseg.shape)

    def getDataFrame(self):
        print("Getting data frame")