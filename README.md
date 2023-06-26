# pitching-biomechanics
All data acquired through Driveline Baseball's github repository "openbiomechanics" @drivelineresearch by Kyle Boddy and Kyle Wasserberger.

## throwing velocity neural network
Using PyTorch linear layers and ReLU activation functions, a neural network is used to predict throwing velocity and perform feature analysis. Through hyperparameter tuning, the model is able to on average be within 6mph of the ground truth value despite only ~400 training samples. The feature analysis displays what biomechanical features are most and least predictive of velocity. All of this is valuable context when allocating training economy and selecting specific mechanics to work on.

## biomech scorer
A Dash dashboard that takes an input file of the appropriate length and type (1x81 and csv/xls) which is the same as one row of the poi_metrics.csv data. It will visualize weighted percentile values with a polar chart for 6 different areas: arm action, arm velos, rotation, pelvis, lead leg block, and cog. Each column was manually put into one of the six categories and the individual's percentile rankings are combined with column Rsquared values with pitch velocity to determine the six scores. Additional code will export a sorted list of an athlete's percentile rankings for each metric, as well as print a composite (total) biomechanics score.

![biomech_scorer](https://user-images.githubusercontent.com/96801448/219224868-fd0fc58b-af8c-4271-8b0c-6b0f92258fe9.jpg)

## biomech vis:
A Dash dashboard for visualizing and exploring the relationships between different pitching biomechanical metrics and throwing velocity. Helpful in discerning R^2 values and relationship trends, as well as the population distribution of different metrics.

https://user-images.githubusercontent.com/96801448/211731114-fbe28913-286e-4cb4-9c78-05e27faf4649.mp4

## velocity predictor (newer & better model above but without UI):
A Dash dashboard that allows the user to predict pitching velocity by inputting five biomechanical metrics:
1. shoulder horizontal abduction at foot plant
2. max shoulder external rotation
3. max shoulder internal rotational velocity
4. max torso rotational velocity
5. rotational hip-shoulder separation at foot plant

These were selected by determining Rsquared coefficients for each metric with pitch velocity. I chose only metrics that involved the speed or positioning achieved by the thrower due to their ability to be trained rather than metrics like energy flow or joint moments which I deemed to be more of a product of those positions and velocities. I also wanted to make sure my selections covered different areas of the throw: shoulder, torso, hips.

https://user-images.githubusercontent.com/96801448/212527424-d88d4e3c-2a3c-4d46-9fe5-636d3a0e408d.mp4
