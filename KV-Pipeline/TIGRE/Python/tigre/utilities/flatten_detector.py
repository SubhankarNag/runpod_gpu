import numpy as np

def find_angles(geo):

    arclength = np.deg2rad(geo.arcLength) # has to be 2 * 32.9 degrees, changed to radians
    orig_num_detectors = geo.nDetectorCurved[1] # has to be 480
    pixel_arclength = arclength / orig_num_detectors
    orig_num_scintillators = geo.nScintillators
    orig_sensors_per_scintillator = geo.sensorsPerScintillator[1] # along column, has to be 16
    orig_scintillator_size = geo.scintillatorSize[1] # along columns, has to be 40.22  

    if orig_num_detectors % 2:
        detector_size = np.tan(pixel_arclength) * geo.DSD
    else:
        detector_size = 2 * np.tan(pixel_arclength / 2) * geo.DSD # has to be 5.122

    total_detector_size = 2 * np.tan(arclength / 2) * geo.DSD # has to be 1348.5508

    if orig_num_detectors % 2:
        detectors = np.arrange(detector_size, (total_detector_size / 2) + 0.01, detector_size)   # added 0.01 for endcase problem
        detectors = np.concatenate([-np.flip(detectors), [0], detectors])                     #### Have to discuss end-case issue 
    else:
        detectors = np.arange(detector_size / 2, (total_detector_size / 2) + 0.01, detector_size)
        detectors = np.concatenate([-np.flip(detectors), detectors])

    flat_num_detectors = len(detectors)
    final_angles = np.arctan2(detectors, geo.DSD)

    orig_angles = np.zeros((orig_num_detectors, 1))
    for i in range(orig_num_scintillators):
        theta = np.arctan2(geo.pScintillator[i][0], geo.pScintillator[i][1])
        alpha = np.arctan2(orig_scintillator_size / 2, np.sqrt((geo.pScintillator[i][0] ** 2) + (geo.pScintillator[i][1] ** 2)))
        # Below assumes scintillator scitinlator is curved. Seems an approximation
        orig_angles[i * orig_sensors_per_scintillator:(i+1) * orig_sensors_per_scintillator] = np.linspace(theta - alpha, theta + alpha, orig_sensors_per_scintillator).reshape(-1, 1)

    return orig_angles, final_angles, flat_num_detectors, detector_size


def flatten_detector(proj, geo):

    orig_angles, final_angles, flat_num_detectors, detector_size = find_angles(geo)

    flattened_proj = np.zeros((proj.shape[0], proj.shape[1], flat_num_detectors))
    for i in range(proj.shape[0]):
        for r in range(proj.shape[1]):
            # has better intrpolates in scipy, not sure below works correctly
            flattened_proj[i, r, :] = np.interp(np.squeeze(final_angles), np.squeeze(orig_angles), np.squeeze(proj[i, r, :]))

    geo.nDetector = np.array((flattened_proj.shape[1], flattened_proj.shape[2]))
    geo.dDetector = np.array((geo.dDetectorCurved[0], detector_size))
    geo.sDetector = geo.nDetector * geo.dDetector

    return flattened_proj


def unflatten_detector(flat_proj, geo):

    orig_angles, final_angles, _, _ = find_angles(geo)
    unflattened_proj = np.zeros((flat_proj.shape[0], flat_proj.shape[1], geo.nDetectorCurved[1]))
    for i in range(flat_proj.shape[0]):
        for r in range(flat_proj.shape[1]):
            # has better intrpolates in scipy, not sure below works correctly
            unflattened_proj[i, r, :] = np.interp(np.squeeze(orig_angles), np.squeeze(final_angles), np.squeeze(flat_proj[i, r, :]))

    return unflattened_proj