# udcarlab_summer2025_stopsign
Stop Sign Detection and Right-of-Way Decision System in ROS 2
Dean Yao

1 Overview
This project implements a ROS 2 perception and decision making node for handling stop sign scenarios in an autonomous driving simulation.
The system integrates camera based object detection (via YOLO), stop detection logic, and a right of way manager to decide when it is safe for the vehicle to proceed.

The pipeline goes as follows:
Image subscriber receives RGB images from the ego vehicle’s camera.
YOLO detection identifies stop signs and other vehicles.
Stop condition check ensures the vehicle is fully stopped before evaluating cross traffic.
The Right of Way Manager class tracks other vehicles in the intersection ROI and decides when it is safe to go.

2 ROS2 Node Architecture
The system is implemented as a single ROS 2 node, StopSignManager, with helper classes for detection parsing, object tracking, and right of way management.

Subscriptions
Camera (sensor_msgs/Image)
Topic: /sensing/camera/camera0/image_rect_color
QoS: Best Effort, Keep Last (depth=10)
Image encoding: 8UC4 (BGRA → converted to BGR for YOLO)
LiDAR (sensor_msgs/PointCloud2)
Topic: /sensing/lidar/top/outlier_filtered/pointcloud
Currently used for logging, potential future integration is planned.

3 Helper Classes and Methods

get_detections_as_list() method
Parses Ultralytics YOLO results into an array: [x1, y1, x2, y2, conf, class_id, label] where (x1, y1) and (x2, y2) are the bottom left and top right corner of the bounding box respectively, conf is the confidence score given by YOLO, class_id, is the class ID in the COCO dataset, and label is the class in the COCO dataset.


Tracker Class

Init
next_id → a unique index that will be assigned to the current detection and incremented for the next detection.
tracks → a dictionary mapping each detection's unique index to their most recent detected centroid.
max_distance → constant value. Two centroids across frames are considered the same detection if their euclidean distance is less than max_distance

centroid method
Gets centroid of detection.
Input → bounding box coordinates of detection
Output → tuple of centroid of detection (x, y)

update method
Takes in an array of detections in the format [x1, y1, x2, y2, conf, class_id, label] and assigns a persistent track ID to each detection based on centroid proximity to detections from previous frames.

Input → detections: list of [x1, y1, x2, y2, conf, class_id, label]
Output → new list of [(cx,cy), conf, class_id, label, track_id]

Logic:
Extract centroids. For each detection, compute the centroid from the bounding box coordinates using the centroid method.
Track ID assignment. For each centroid, compare it against all existing tracked centroids from previous frames (self.tracks). If the Euclidean distance between a new centroid and an existing centroid is less than max_distance, reuse that track’s ID. If no match is found, assign a new ID (self.next_id) and increment it.
Update track dictionary. Store the latest centroid for each track ID in self.tracks so that the next frame can match against the updated positions.
Output formatting. Return a new list of tracked detections in the format [(cx, cy), conf, class_id, label, track_id].


RightOfWayManager Class

Init
roi → rectangular Region Of Interest for the intersection (x_min, y_min, x_max, y_max).
max_missed_frames → how many consecutive frames a previously seen vehicle can be absent from the ROI before we assume it has cleared the intersection.
seen_init_count → frame counter for initial observation window. Vehicles seen in this window have priority.
seen_initially → boolean flagging the initial observation window.
priority_ids → set of track IDs (cars/buses/trucks) present in the ROI during the initial window. These are the vehicles we must yield to.
priority_missed → dict mapping track_id to missed_frame_count 
tracker → instance of the Tracker class used to assign stable IDs to detections across frames.

InROI method
Checks whether a tracked object’s centroid lies inside the ROI.
Input → centroid: (x, y)
Output → bool (True if inside ROI, False otherwise)

process_frame method
Takes in one YOLO result, updates tracking, maintains priority vehicles, and decides if the intersection is clear.
Input → res (YOLO result for the current frame)
Output → bool (True = SAFE TO GO, False = DO NOT GO)

Logic:
Parse and Track
Convert res to detections via get_detections_as_list
Run self.tracker.update(detections) to get [(cx, cy), conf, class_id, label, track_id].
Collect priority vehicles via initial window
While not seen_initally
For each tracked entry, if class_id in {2, 5, 7} (car, bus, truck) and InROI(centroid), add its track_id to priority_ids.
Initialize each priority_missed[track_id] = 0.
After seen_init_count exceeds a small threshold of initial frames (e.g., 3 frames), set seen_initially = True.
Until then, return False (still collecting who has the right of way).
Post initial window monitoring
Create still_in_ROI set: all track_id in priority_ids that are currently inside ROI.
For every track_id in priority_ids but not in still_in_ROI, increment priority_missed[track_id] by 1.
For every track_id in still_in_ROI, reset priority_missed[track_id] = 0.

Clear vehicles that have left the ROI
For each track_id in priority_ids, if priority_missed[track_id] > max_missed_frames, remove it from priority_ids and delete its entry in priority_missed.
Removing from priority_ids means that the vehicle has left the intersection
Decision
If priority_ids is now empty → all initially present vehicles that had priority over the ego vehicle have cleared, return True (SAFE TO GO).
Otherwise → there exists vehicles that still have priority, return False.

4 Main ROS 2 Node

StopSignManager Class
ROS2 node that detects stop signs from camera images, determines if the ego vehicle has stopped, and checks intersection clearance using RightOfWayManager.

Init
Sets up parameters, YOLO detection model, right-of-way logic, and ROS2 subscribers.
latest_image / latest_lidar → store the most recent camera and LiDAR messages.
Declared ROS parameters:
yolo.model → YOLO model file path
bridge → CvBridge instance for ROS to OpenCV image conversion.
yolo → Ultralytics YOLO object detection model for stop sign and vehicle detection.
romgr → RightOfWayManager instance for intersection clearance logic
Stop detection tuning:
prev_sign_w, prev_sign_h → store last observed width/height of stop sign bounding box.
sign_stable_frames → counter of consecutive “stable size” frames.
stop_hold_frames → how many stable frames before declaring the vehicle stopped.
inc_percentage → max allowed relative change in width/height for “stable” classification.
min_sign_conf → YOLO confidence threshold for a valid stop sign detection.
ROS2 QoS settings (BEST_EFFORT, KEEP_LAST, depth=10) for camera/LiDAR topics.
Subscriptions:
/sensing/camera/camera0/image_rect_color → handled by image_callback.
/sensing/lidar/top/outlier_filtered/pointcloud → handled by lidar_callback.

image_to_bgr method
Converts an 8UC4 ROS Image message to a BGR OpenCV image.
largest_stop_sign method
Finds the single largest stop sign (class ID: 11) with confidence above min_sign_conf.
Input → list of YOLO detections in [x1, y1, x2, y2, conf, cls, label] format.
Output → (x1, y1, x2, y2, conf) for largest-area stop sign, or None if none qualify.

stopped method
Determines if the ego vehicle has been stationary relative to the stop sign for at least stop_hold_frames.
Logic:
Get the largest stop sign via largest_stop_sign.
If no sign found → reset previous size, counter, and return False.
Compute current width (w) and height (h).
If no previous width/height recorded → initialize and return False.
Compute relative change in width (rel_dw) and height (rel_dh) compared to previous frame.
If both changes are below inc_percentage → increment sign_stable_frames, else reset counter.
Update stored width/height.
Return True if sign_stable_frames >= stop_hold_frames.

image_callback method
Main decision loop triggered by each incoming camera frame.
Store the image.
Convert it to BGR (image_to_bgr).
Run YOLO detection → parse results into a list via get_detections_as_list.
If a stop sign exists (largest_stop_sign is not None):
If stopped(detections) is True:
Log “STOP SIGN DETECTED! CHECKING INTERSECTION” and call romgr.process_frame to see if it’s safe to go.
Log “SAFE TO GO” if True, else “DO NOT GO”.
Else → log “SIGN PRESENT, NOT FULLY STOPPED YET”.
If no stop sign detected → log “NO STOP SIGN DETECTED, CONTINUING”.

