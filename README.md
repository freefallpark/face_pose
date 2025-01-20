# Software Dependencies Summary
- glog - for logging.
- FastDDS - for IPC. 
- OpenCV - for image processing
- depthai - camera handling and image processing (luxonis)

# Inputs:
- BaseCamera Frames
- Transform from world frame to camera frame

# Outputs:
- Pose(s) of detected faces

# IPC
Project uses FastDDS for inter-process communication. For detailed information about the Face Pose implementation of
FastDDS, see [Face Pose IPC](ipc/README.md).

# Pose Estimation
Looks for faces in the camera frame and produces a list of poses for each unique face. For detailed information, see
[Pose Estimation Library](pose_estimation/README.md)