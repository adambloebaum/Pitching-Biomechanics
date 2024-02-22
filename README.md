# biomech

a collection of biomechanics related projects, some of which use the openbiomechanics project data (see .bib)

## dashboards

### biomech viewer

a dashboard for viewing statistical relationships between biomechanical features and pitch velocity. interactive dropdown for selecting the y-axis of the scatter plot, includes calculated trend line and rsquared value. interactive dropdown for selecting the x-axis of the histogram

### composite score dash

a dashboard for viewing biomechanic composite score groups. broken down into arm action, arm velos, torso, pelvis, lead leg block, and cog (center of gravity). users can upload 1 pitch of poi_metrics (can get from OBP) and view the percentile ranking for each group on a radial plot. feature importance is weighted using rsquared values with pitch velocity

## deeplearning

a pytorch neural network for predicting pitch velocity using biomechanics data. new features relative to body weight are engineered, and all features are standard scaled. an 80-10-10 train-validation-test data split is utilized. the regression model is comprised of linear layers of various size accompanied by ReLU activation functions and 2 dropout layers. gpu acceleration is enabled to expedite the training process. mean squared error loss function and adam optimizer are used. early stopping, dropout, and a learning rate scheduler are implemented to prevent overfitting. training loss and validation loss are displayed during training/validation, training loss over epochs is plotted, and MSE is displayed following the testing loop

model parameters are optimized through both trial and error and utilizing the optuna framework for hyperparameter tuning. model architecture, loss function, and optimizer are determined through trial and error. learning rate, batch size, and dropout probability are optimized through an optuna study. pruning is utilized to optimize computation time and gpu resources

local interpretable model-agnostic explanations (LIME) package is used to approximate the neural network with a local, interpretable model. the first value in the test set is used to produce a local explainer model illustrating what went into the model's prediction

a test mean squared error of __10.79__ was achieved, making the model's predictions on average off by __3.28mph__. the remaining error can be attributed to physical performance attributes like strength and power output as well as the variability involved in the act of pitching