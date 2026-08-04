"""
Custom TIGRE Geometry Module
Defines the physical layout of the scanner (X-ray source, Curved Detector, Voxels).
Inherits from TIGRE's base Geometry class.
"""
from __future__ import division
import numpy as np
from tigre.utilities.geometry import Geometry

class CTGeometry(Geometry):
    def __init__(self, config_geom):
        super().__init__()
        
        # Distance configuration
        self.DSD = config_geom['dsd']
        self.DSO = config_geom['dso']
        
        # Detector configuration
        self.nDetectorCurved = np.array(config_geom['ndetectorcurved'])
        self.dDetectorCurved = np.array(config_geom['ddetectorcurved'])
        # Total size of detector is pixel count * physical pixel size
        self.sDetectorCurved = self.nDetectorCurved * self.dDetectorCurved
        self.arcLength = config_geom['arc_length']
        
        # Scintillator Positions
        if config_geom['pscintillator_path'] != 'default':
            self.pScintillator = np.load(config_geom['pscintillator_path'])
        else:
            # Fallback hardcoded module coordinates relative to source
            self.pScintillator = np.array([
                [-554.58, 893.31], [-519.97, 913.94], [-484.59, 933.22], [-448.5, 951.13],
                [-411.78, 967.63], [-374.38, 982.73], [-336.47, 996.38], [-298.7, 1008.56],
                [-259.22, 1019.26], [-219.99, 1028.45], [-180.44, 1036.14], [-140.62, 1042.3],
                [-100.59, 1046.93], [-60.4, 1050.13], [-20.13, 1051.57], [20.13, 1051.57],
                [60.4, 1050.03], [100.59, 1046.93], [140.62, 1042.3], [180.44, 1036.14],
                [219.99, 1028.45], [259.22, 1019.26], [298.7, 1008.56], [336.47, 996.38],
                [374.38, 982.73], [411.78, 967.63], [448.5, 951.13], [484.59, 933.22],
                [519.97, 913.94], [554.58, 893.31]
            ])
            
        self.nScintillators = config_geom['nscintillators']
        self.sensorsPerScintillator = np.array(config_geom['sensorsperscintillator'])
        self.scintillatorSize = np.array(config_geom['scintillatorsize'])
        
        # Voxel grid configuration
        self.dVoxel = np.array(config_geom['dvoxel'])
        self.sVoxel = np.array([config_geom['svoxel_x'], config_geom['svoxel_y'], config_geom['svoxel_z']])
        
        # Compute number of voxels by dividing total physical size by voxel resolution
        self.nVoxel = np.array([
            int(config_geom['svoxel_x'] / self.dVoxel[0]), 
            int(config_geom['svoxel_y'] / self.dVoxel[1]), 
            int(config_geom['svoxel_z'] / self.dVoxel[2])
        ])
        self.sVoxel = self.nVoxel * self.dVoxel
        
        # Default offsets and modes (TIGRE requirements)
        self.offOrigin = np.array((0, 0, 0))
        self.offDetector = np.array((0, 0))
        self.rotDetector = np.array((0, 0, 0))
        self.accuracy = 0.5 
        self.mode = "cone"
        self.filter = None
        
        # Placeholders populated later during TIGRE detector flattening
        self.nDetector = None
        self.dDetector = None
        self.sDetector = None

    def update_for_angles(self, num_angles, shift_z, disp):
        """Dynamically builds the offset array based on the number of projection angles."""
        self.offOrigin = np.zeros((num_angles, 3))
        self.offOrigin[:, 2] = shift_z
        self.offOrigin[:, 0] = disp