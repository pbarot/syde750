<<<<<<< HEAD
=======
syde750


## Humanoids project - Speech Identification and Speaker Tracking on REEM-C
### Code breakdown

#### Vision functionality and ROS Interfacing

*class_face.py* - Class for faces to compute their own metrics and process features, required results and establish own thresholds

*feature_extraction.py* - A set of numerous utilities to generate required features, process images and estimate presence of speech

*two_person_case.py* - Interfacing with RealSense to generate depth and colour stream, identify faces, predict individual speech presence and publish relevant yaw angles and classification results to ROS network

*send_head_torso.py* - Subscribe to published yaw angle and convert to required head and torso yaw angles given previous yaw angle

*send_tup_tdown.py* - Subscribe to published speech identification signal and add trajectories for corresponding thumb gesture. *send_double_t.py* handles this for both arms in the 
two person scenario

#### Launch files

*full_demo.launch* - Run single speaker scenario code while launching Gazebo and the custom head/torso controller

=======

*full_func.launch* - Run double speaker scenario while launching custom head/torso controller. Gazebo should be run first before this

*head_torso_control.launch* - Run custom head/torso controller, using config from .yaml file


#### Custom config

*head_torso_control.yaml* - Config for custom head/torso controller



#### misc folder: code used to test some functionality, visualize, generate some demo images/save data for visualization later. Includes similar algorithms developed with just webcam


*full_func.launch* - Run double speaker scenario while launching custom head/torso controller. Gazebo should be run first before this

*head_torso_control.launch* - Run custom head/torso controller, using config from .yaml file


#### Custom config

*head_torso_control.yaml* - Config for custom head/torso controller


#### misc folder: code used to test some functionality, visualize, generate some demo images/save data for visualization later. Includes similar algorithms developed with just webcam
#### includes algo_exp.html where some initial exploration and algorithm development was performed



