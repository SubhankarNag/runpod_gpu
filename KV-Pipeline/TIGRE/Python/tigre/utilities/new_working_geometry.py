from __future__ import division

import numpy as np
from tigre.utilities.geometry import Geometry


class newConeGeometryWorking(Geometry):
    def __init__(self, nVoxel_x=1000, nVoxel_y=620, nVoxel_z=420, size_x=1000, size_y=620, size_z=420):

        Geometry.__init__(self)
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = 1050.0  # Distance Source Detector      (mm)
        self.DSO = 576.28  # Distance Source Origin        (mm)
        # Detector parameters (Curved detector)
        self.nDetectorCurved = np.array((32, 480))  # (V,U) number of pixels        (px)
        self.dDetectorCurved = np.array((2.75, 2.5))  # size of each pixel            (mm)
        self.sDetectorCurved = self.nDetectorCurved * self.dDetectorCurved  # total size of the detector    (mm)
        self.arcLength = 2 * 32.9 # detector plate angle at source (degrees)
        # Below must be updated in flatten detector
        self.nDetector = None
        self.dDetector = None
        self.sDetector = None
        # Image parameters
        self.nVoxel = np.array((nVoxel_x, nVoxel_y, nVoxel_z))  # number of voxels              (vx)
        self.sVoxel = np.array((size_x, size_y, size_z))  # total size of the image       (mm)
        self.dVoxel = self.sVoxel / self.nVoxel  # size of each voxel            (mm)
        # Offsets
        self.offOrigin = np.array((0, 0, 0))  # Offset of image from origin   (mm)  #has to change 
        self.offDetector = np.array((0, 0))  # Offset of Detector            (mm)
        self.rotDetector = np.array((0, 0, 0))
        # Auxiliary
        self.accuracy = 0.5  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = "cone"  # parallel, cone                ...
        self.filter = None
        # Scintilator positions
        self.pScintillator = np.array([     # position of scintilator centers wrt source (mm)
        [-554.58, 893.31],
        [-519.97, 913.94],
        [-484.59, 933.22],
        [-448.5, 951.13],
        [-411.78, 967.63],
        [-374.38, 982.73],
        [-336.47, 996.38],
        [-298.7, 1008.56],
        [-259.22, 1019.26],
        [-219.99, 1028.45],
        [-180.44, 1036.14],
        [-140.62, 1042.3],
        [-100.59, 1046.93],
        [-60.4, 1050.13],
        [-20.13, 1051.57],
        [20.13, 1051.57],
        [60.4, 1050.03],
        [100.59, 1046.93],
        [140.62, 1042.3],
        [180.44, 1036.14],
        [219.99, 1028.45],
        [259.22, 1019.26],
        [298.7, 1008.56],
        [336.47, 996.38],
        [374.38, 982.73],
        [411.78, 967.63],
        [448.5, 951.13],
        [484.59, 933.22],
        [519.97, 913.94],
        [554.58, 893.31],
        ])
        self.nScintillators = 30
        self.sensorsPerScintillator = np.array((32, 16))  # (V,U) number of pixels        (px)
        self.scintillatorSize = np.array((88.0, 40.22)) # size of each scintillator (mm)