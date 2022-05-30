
# Multi-Object Tracking with Deep Learning
This repository consists of scripts and frameworks used to develop, analyze and evaluate deep learning frameworks to describe objects. The work has tested out three different variations. 
1. With Masked Out Background (MOB) 
`scripts/tracking_ini.py` contains the script to run the tracker with evaluation. The tracker is implemented in `utilities/tracker/tracker.py´
2. SuperPoint and SuperGlue. 
`scripts/tracking_superpoint.py` contains the script to run the tracker with evaluation. The tracker is implmented in ´utilities/tracker/tracker_superpoint.py´
3. MOB combined with SuperPoint and SuperGlue
`scripts/tracking.py´ contains the script to run the tracker with evaluation. The tracker is implmented in ´utilities/tracker/tracker_combined.py´

## Getting Started
The code is implemented in the Emily v.1.8.0, which maintains a stable docker environment for development. To run, install Emily https://github.com/amboltio/emily-cli (Release-V. 2.0.0 should be compatible).
run `emily open thesis´ to run the project.
