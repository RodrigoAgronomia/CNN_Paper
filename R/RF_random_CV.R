# General package to use and compare different ML methods:
library(mlr)

# Fast Random Forest implementation:
library(ranger)

# Read the training set:
training_set = readRDS('data/training.rds')

# Read the test set:
test_set = readRDS('data/test.rds')

# Make the regression task, using yield as the target variable,
# and all others ("elev", "N", "SR", "EC", "soil") as predictors:
regr.task = makeRegrTask(data = training_set, target = "yield")

# Create a resample strategy to adjust Hyperparameters,
# this should take into accont the spatial correlation of the neighbor points:
rdesc = makeResampleDesc("RepCV", fold = 5, reps = 5)

# Create the RF learner, with some optimized hyperparameters:
learner.rf = makeLearner("regr.ranger",
                         num.trees = 500,
                         min.node.size = 25)

# Apply the k-fold CV to fit the models to each subset:
mod = resample(learner.rf, regr.task, rdesc)

# Save the predictions from the k-fold CV:
train_pred = data.frame(mod$pred)

# Compare the predictions with the observed values in the training set:
plot(truth ~ response, train_pred, asp = 1, pch = '.')
cor(train_pred$truth, train_pred$response)

# Fit a new model to all the training data:
mod_fitted = train(learner.rf, regr.task)

# Use the model to predict in the test set:
test_pred = predict(mod_fitted, newdata = test_set)$data

# Compare the predictions with the observed values in the test set:
plot(truth ~ response, test_pred, asp = 1, pch = '.')
cor(test_pred$truth, test_pred$response)


