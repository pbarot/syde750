# syde750
Humanoids project - Speech Identification As Reflected by Robotic Actuation

## Task breakdown
1. ROS based infrastructure to capture camera feed, apply DLIB (or a relevant facial detector) and publish the extracted landmarks and coordinates to a ROS topic 

2. Either find or generate our own dataset of speech/no speech given a video of a single speaker. It should include many periods of naturally waiting and watching the camera, and periods of 
utterances/sentences of a few seconds long

3. Develop a signal processing approach to classify mouth opening distance signals as 1 (speech) or 0 (no speech). Most likely based on breaking signals into frames and classifying each one
Consider

- Significant RMS/Power changes in the input frames

- Rate of change in input frames

- Normalizing by length of the face to account for speaker's distance to camera

4. Publish this binary signal on the network. Allow for a joint trajectory controller to receive this message and activate the thumbs up/down state

5. Design actuations to achieve the thumbs up/down state 

6. Use one facial landmark to estimate yaw and pitch required to center the speaker's face. Send these estimates to the head and torso (more on this later)



## Relevant docs/links

- Auditory and Visual Speech Relationships https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000436
