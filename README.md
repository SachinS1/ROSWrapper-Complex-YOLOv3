# ROS Wrapper for ComplexYOLOv3 Implementation

ComplexYOLOv3 implementation was forked from [this repository](https://github.com/ghimiredhikura/Complex-YOLOv3).

## Package Installation
```bash
cd catkin_ws/src/
git clone https://github.com/SachinS1/ROSWrapper-Complex-YOLOv3.git
cd ../..
catkin_make #or catkin_build depending on how you setup your directory
```

After compiling the package, run the following to run the ROS node.
```bash
roslaunch complexyolov3_ros complexyolov3.launch
```

Make sure to specify the appropriate LiDAR topic in the launch file (`point_cloud_topic` parameter).
