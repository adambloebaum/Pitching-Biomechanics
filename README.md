# pitching-biomechanics
All data acquired through Driveline Baseball's github repository "openbiomechanics" @drivelineresearch by Kyle Boddy and Kyle Wasserberger.

## biomech scorer
A Dash dashboard that takes an input file of the appropriate length and type (1x81 and csv/xls) which is the same as one row of the poi_metrics.csv data. It will visualize weighted percentile values with a polar chart for 6 different areas: arm action, arm velos, rotation, pelvis, lead leg block, and cog. Each column was manually put into one of the six categories and the individual's percentile rankings are combined with column Rsquared values with pitch velocity to determine the six scores. Additionally code will export a sorted list of an athlete's percentile rankings for each metric, as well as print a composite (total) biomechanics score.

![biomech_scorer](https://user-images.githubusercontent.com/96801448/219224868-fd0fc58b-af8c-4271-8b0c-6b0f92258fe9.jpg)

## biomech vis:
A Dash dashboard for visualizing and exploring the relationships between different pitching biomechanical metrics and throwing velocity. Helpful in discerning R^2 values and relationship trends, as well as the population distribution of different metrics.

https://user-images.githubusercontent.com/96801448/211731114-fbe28913-286e-4cb4-9c78-05e27faf4649.mp4

## velocity predictor:
A Dash dashboard that allows the user to predict pitching velocity by inputting five biomechanical metrics:
1. shoulder horizontal abduction at foot plant
2. max shoulder external rotation
3. max shoulder internal rotational velocity
4. max torso rotational velocity
5. rotational hip-shoulder separation at foot plant

These were selected by determining Rsquared coefficients for each metric with pitch velocity. I chose only metrics that involved the speed or positioning achieved by the thrower due to their ability to be trained rather than metrics like energy flow or joint moments which I deemed to be more of a product of those positions and velocities. I also wanted to make sure my selections covered different areas of the throw: shoulder, torso, hips.

Rsquared values:

('shoulder_horizontal_abduction_fp', 0.14310639482494072)

('max_shoulder_external_rotation', 0.11000169464538753)

('max_shoulder_internal_rotational_velo', 0.1057783536315822)

('max_torso_rotational_velo', 0.10245061616290904)

('rotation_hip_shoulder_separation_fp', 0.0967202287144392)

A deep learning neural network is used to establish weight and bias for each of the four metrics. Mean absolute error is used to plot loss and over 200 epochs it is minimized to just under 2.5 mph.

https://user-images.githubusercontent.com/96801448/212527424-d88d4e3c-2a3c-4d46-9fe5-636d3a0e408d.mp4
