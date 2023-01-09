# pitching-biomechanics
All data acquired through Driveline Baseball's github repository "openbiomechanics" @drivelineresearch by Kyle Boddy and Kyle Wasserberger.
## biomech vis:
A Dash dashboard written in Python for visualizing and exploring the relationships between different pitching biomechanical metrics and throwing velocity. Helpful in discerning R^2 values and relationship trends, as well as the population distribution of different metrics.

## velocity predictor:
A Dash dashboard written in Python that allows the user to predict pitching velocity by inputting five biomechanical metrics:
1. shoulder horizontal abduction at foot plant
2. max shoulder external rotation
3. max shoulder internal rotational velocity
4. max torso rotational velocity
5. rotational hip-shoulder separation at foot plant

These were selected by determining Rsquared coefficients for each metric with pitch velocity. I chose only metrics that involved the speed or positioning achieved by the thrower due to their increased trainability compared to metrics like energy flow or joint moments which I deemed to be more of a product of "good" pitching mechanics. I also wanted to make sure my selections covered different areas of the throw: shoulder, torso, hips.

Rsquared values:

('shoulder_horizontal_abduction_fp', 0.14310639482494072)

('max_shoulder_external_rotation', 0.11000169464538753)

('max_shoulder_internal_rotational_velo', 0.1057783536315822)

('max_torso_rotational_velo', 0.10245061616290904)

('rotation_hip_shoulder_separation_fp', 0.0967202287144392)

A deep learning neural network is used to establish weight and bias for each of the four metrics. Mean absolute error is used to plot loss and over 200 epochs it is minimized to just under 2.5 mph.

![mse_loss_plot](https://user-images.githubusercontent.com/96801448/211247335-ddec7dcc-1641-4a3e-9682-f5c8ffdd4dda.png)
