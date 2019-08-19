# General package to use and compare different ML methods:
library(mlr)

# Fast Random Forest implementation:
library(ranger)

# To plot the results:
library(ggplot2)

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



# Create the linear model learner:
learner.lm = makeLearner("regr.lm")

# Apply the k-fold CV to fit the models to each subset:
mod_lm = resample(learner.lm, regr.task, rdesc)

# Save the predictions from the k-fold CV:
train_pred = data.frame(mod_lm$pred)

# Compare the predictions with the observed values in the test set:
print(paste0('Correlation: ', round(cor(train_pred$truth, train_pred$response), 2)))
print(paste0('RMSE: ', round(measureRMSE(train_pred$truth, train_pred$response), 2)))
ggplot(train_pred, asp = 1, aes(x = response, y = truth)) + geom_point()
ggsave('figures/train_lm.png')


# Create the SVM learner:
learner.svm = makeLearner("regr.svm")

# Apply the k-fold CV to fit the models to each subset:
mod_svm = resample(learner.svm, regr.task, rdesc)

# Save the predictions from the k-fold CV:
train_pred = data.frame(mod_svm$pred)

# Compare the predictions with the observed values in the test set:
print(paste0('Correlation: ', round(cor(train_pred$truth, train_pred$response), 2)))
print(paste0('RMSE: ', round(measureRMSE(train_pred$truth, train_pred$response), 2)))
ggplot(train_pred, asp = 1, aes(x = response, y = truth)) + geom_point()
ggsave('figures/train_svm.png')


# Create the RF learner, with some optimized hyperparameters:
learner.rf = makeLearner("regr.ranger",
                         num.trees = 100,
                         min.node.size = 25)

# Apply the k-fold CV to fit the models to each subset:
mod_rf = resample(learner.rf, regr.task, rdesc)

# Save the predictions from the k-fold CV:
train_pred = data.frame(mod_rf$pred)

# Compare the predictions with the observed values in the test set:
print(paste0('Correlation: ', round(cor(train_pred$truth, train_pred$response), 2)))
print(paste0('RMSE: ', round(measureRMSE(train_pred$truth, train_pred$response), 2)))
ggplot(train_pred, asp = 1, aes(x = response, y = truth)) + geom_point()
ggsave('figures/train_rf.png')



# Fit a new model to all the training data:
mod_fitted = train(learner.rf, regr.task)

# Use the model to predict in the test set:
test_pred = predict(mod_fitted, newdata = test_set)$data

# Compare the predictions with the observed values in the test set:
print(paste0('Correlation: ', round(cor(test_pred$truth, test_pred$response), 2)))
print(paste0('RMSE: ', round(measureRMSE(test_pred$truth, test_pred$response), 2)))
ggplot(test_pred, asp = 1, aes(x = response, y = truth)) + geom_point()
ggsave('figures/test.png')
