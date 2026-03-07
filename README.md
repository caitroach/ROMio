# ROMio - a markerless range of motion measurment system for physicians

In our 2026 QMIND Healthcare project, we used an Intel RealSense D435 camera with depth-sensing technology. It combines RGB cameras with infrared stereo depth sensors to produce a hybrid image of both colour and depth. These cameras are widely used in tasks involving facial recognition. 

We are using this camera in combination with MediaPipe's pretrained models to compute real-time angles of rotation in a subject's shoulders, exploring the feasibility of such a system in orthopedic ROM assessment. This is a work in progress!

## how it works 
We're using MediaPipe PoseLandmarker to run a neural net on each video frame to detect 33 body landmarks, which are normalized as (x, y, z) coordinates. The x and y coordinates are scaled from 0 to 1. The z coordinate is a rough estimate using the webcam until the Intel RealSense depth camera is activated. Then the program replaces that z coordinate with the true measured depth in meters from the IR depth sensor, which is highly accurate. 

Using measured 3D coordinates, we project vectors representing shoulders and upper arms. We compute clinically relevant angles of rotation using vector math. Anterior rotation is calculated by projecting the upper arm vector onto the forward axis using arcsin. Interior rotation projects the forearm and a reference direction onto the plane perpendicular to the upper arm, then measures the angle between them around the upper arm axis. These computational techniques were taken from Mallare *et. al*. 

The app is presented as a live feed with a skeleton overlay so users can view the measurements in real-time. The UI was designed with CUCAI 2026 in mind, so it was made to be visually appealing for the purpose of our live demo. To reduce jittering, each displayed angle reading is averaged over the last 8 frames. 

Our aim is to provide an easy, accurate ROM measurement for physicians, proving more convenient than goniometers.

## team members
Team lead: Cait Roach

Design team: Hargun Kour, Basma Azeem, Augustine Osezua, Gavin Tan, Leif Hill, Nicholas Irons, Tharunika Gnaneshan

## related work and citations
we used a lot of the math and code from these repos/studies, so please check them out too: 
- https://arxiv.org/pdf/2310.07322 
- https://www.researchgate.net/publication/322815664_Sitting_posture_assessment_using_computer_vision 
- https://github.com/ashita03/Bone-Fracture-Detection 
- https://pitthexai.github.io/research.html
- https://www.diva-portal.org/smash/get/diva2:1897521/FULLTEXT02.pdf
- https://github.com/admiralakber/physio-rom
- https://www.mdpi.com/1424-8220/25/3/667
- https://www.mdpi.com/1424-8220/23/14/6445
