#Import library
library(readxl)
library(neuralnet)
library(ggplot2)
library(MLmetrics)
library(keras)

#PART 1
# Definition of the root-mean-square error
rmse <- function(error) {
  return(sqrt(mean(error^2)))
}

# Definition of the mean absolute error
mae <- function(error) {
  return(mean(abs(error)))
}

# Definition of the mean absolute percentage error
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Definition of the symmetric mean absolute percentage error
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted)))*100)
}

# Load the UOW consumption dataset
UOW_consumption_data <- read_xlsx("datasets/uow_consumption.xlsx")

# For the years 2018 and 2019, extrapolate the hourly electricity consumption figures for 20:00.
#   "0.83333333333333337"= 20.00  (20/24 = 0.83333333333333337)
consumption_hourly_20 <- UOW_consumption_data[c("date", "0.83333333333333337")]

# Plot the 20:00 hourly consumption data for 2018 and 2019.
ggplot(UOW_consumption_data, aes(x=date, y=`0.83333333333333337`)) +
  geom_line() +
  labs(title = "Hourly Consumption for 20:00",
       x = "Date",
       y = "Consumption")


# Extract the first 380 samples for training and the rest for testing.
trained_data <- unlist(consumption_hourly_20[1:380, "0.83333333333333337"])
test_dataset <- unlist(consumption_hourly_20[381:nrow(consumption_hourly_20), "0.83333333333333337"])

# The first 380 samples are plotted as training data.
ggplot(data.frame(trained_data), aes(x=1:length(trained_data), y=trained_data)) +
  geom_line() +
  labs(title = "Training Data",
       x = "Sample Number",
       y = "Consumption")


# Determine how many time-delayed inputs there are.
no_inputs <- 60


# Create the input/output matrix for MLP testing and training.
matrix_input_output <- matrix(0, nrow=length(trained_data)-no_inputs, ncol=no_inputs+1)

for (i in 1:(length(trained_data)-no_inputs)) {
  matrix_input_output[i, 1:no_inputs] <- trained_data[i:(i+no_inputs-1)]
  matrix_input_output[i, no_inputs+1] <- trained_data[i+no_inputs]
}

# The input/output matrix be normalized
matrix_input_output <- apply(matrix_input_output, 2, function(x) (x - mean(x)) / sd(x))

# Define the neural network architecture that will be assessed.
structure <- list(
  c(5),
  c(10),
  c(5, 3),
  c(10, 5),
  c(10, 5, 3),
  c(20),
  c(20, 10),
  c(20, 10, 5),
  c(30),
  c(30, 20),
  c(30, 20, 10),
  c(50),
  c(50, 30),
  c(50, 30, 20)
)

result <- list()

for (i in 1:length(structure)) {
  
  # The normalized input/output matrix is used to train the MLP.
  mlp <- neuralnet(V2 ~ ., data=matrix_input_output, hidden=structure[[i]], linear.output=TRUE)
  
  # #Plot the neural network
  # plot(mlp)
  
  # Extract the test data's inputs.
  testing_inputs <- matrix(test_dataset[1:(length(test_dataset)-no_inputs)], ncol=no_inputs, byrow=TRUE)
  
  # Predict the test data's output values.
output_mlp <- predict(mlp, testing_inputs)
  
  # Denormalize the anticipated results.
output_mlp <- (mlp_output * sd(trained_data)) + mean(trained_data)
  
  # Determine the MAE for the actual output values and the predicted output values.
  result_mae <- mae(mlp_output - test_dataset[(no_inputs+1):length(test_dataset)])
  
  cat("the test performances for c(",structure[[i]],")\n")
  
  # Print MAE result
  cat("The MAE for the test data is:", round(result_mae, 2),"\n")
  
  # Determine the RMSE between the output values that were expected and those that were achieved.
result_rmse <- rmse(mlp_output - test_dataset[(no_inputs+1):length(test_dataset)])
  
  # Print the RMSE result
  cat("The RMSE for the test data is:", round(result_rmse, 2),"\n")
  
  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }
  
  # Determine the MAPE using the predicted and actual output values.
result_mape <- mape(test_dataset[(no_inputs+1):length(test_dataset)], output_mlp)
  
  # Print the MAPE result
  cat("The MAPE for the test data is:", round(result_mape, 2),"\n")
  
  # The symmetric mean absolute percentage error (sMAPE) function should be defined.
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }
  
  # Make a SMAPE calculation using the predicted and actual output values.
  result_smape <- smape(test_dataset[(no_inputs+1):length(test_dataset)], mlp_output)
  
  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(result_smape, 2),"\n\n")
  
  # Save the outcome for the current neural network configuration.
  result[[i]] <- c(structure[[i]], result_mae, result_rmse, result_mape, result_smape)
  
}

# Create a data frame of the result
results_dframe <- data.frame(matrix(unlist(result), ncol=5, byrow=TRUE))
colnames(results_dframe) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_dframe)

# Based on MAE and the total number of weights, choose the best one-hidden and two-hidden layer structure.
best_one_hidden <- results_dframe[which.min(results_dframe$MAE & results_dframe$Structure),]
best_two_hidden <- results_dframe[which.min(results_dframe$MAE & results_dframe$Structure),]

# Print the results
cat("Based on the comparison table, the best one-hidden layer neural network structure is",
    paste0("c(", best_one_hidden$Structure, ")"),
    "with a MAE of", best_one_hidden$MAE,
    "and a total number of", best_one_hidden$Structure + 1, "*1+1*1=", best_one_hidden$Structure + 2, "weight parameters.\n")
cat("The best two-hidden layer neural network structure is",
    paste0("c(", best_two_hidden$Structure, ")"),
    "with a MAE of", best_two_hidden$MAE,
    "and a total number of", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 1, "*1+1*1=", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 2, "weight parameters.\n")

#Part 2


# Define a procedure for constructing a neural network model.
build_neural_network <- function(trained_data, test_dataset, input_vars, hidden_structure) {
  
  # Make a neural network formula.
  formula <- paste("hour_20 ~", paste(input_vars, collapse = " + "))
  
  # Create the neural network model utilizing the neuralnet package.
 model_nn <- neuralnetwork(as.formula(formula), trained_data, hidden = hidden_structure)
  
  # The test data for the prediction
matrix_test <- as.matrix(test_dataset[, input_vars, drop = FALSE])
  colnames(matrix_test) <- colnames(trained_data[, input_vars, drop = FALSE])
  
  # Use the neural network model to predict things.
  predictions <- predict(model_nn, matrix_test)
  
  # Bring back the neural network model and its forecasts.
  return(list(model = model_nn, predictions = predictions))
}

  # function to determine various evaluation metrics
  metrics_calculation <- function(actual_val, predicted_val) {
  # Determine the error's root mean square
  rmse <- sqrt(mean((actual_val - predicted_val)^2))
  
  # Make a mean absolute error calculation.
  mae <- mean(abs(actual_val - predicted_val))
  
  # Make a mean absolute percentage error calculation.
  mape <- mean(abs((actual_val - predicted_val) / actual_val)) * 100
  
  # Determine the Symmetric Mean Absolute Percentage Error.
  smape <- mean(abs(actual_val - predicted_val) / (abs(actual_val) + abs(predicted_val)) * 2) * 100
  
  # Provide a list including all the evaluation metrics.
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}


# Add more descriptive column names
colnames(UOW_consumption_data) <- c("date", "hour_18", "hour_19", "hour_20")

# Set up delayed variables for hour_20 with various time delays.
UOW_consumption_data$lag_1 <- lag(UOW_consumption_data$hour_20, 1)
UOW_consumption_data$lag_2 <- lag(UOW_consumption_data$hour_20, 2)
UOW_consumption_data$lag_3 <- lag(UOW_consumption_data$hour_20, 3)
UOW_consumption_data$lag_4 <- lag(UOW_consumption_data$hour_20, 4)
UOW_consumption_data$lag_7 <- lag(UOW_consumption_data$hour_20, 7)

# Remove rows with missing values
UOW_consumption_data <- na.omit(UOW_consumption_data)

# Divide the data into training and test sets according to row index
train <- UOW_consumption_data[1:380,]
test <- UOW_consumption_data[381:nrow(UOW_consumption_data),]

# Create a normalization function that scales data between 0 and 1.
normalized <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Range computation before normalization
before_range <- apply(train[, -1], 2, range)


# Normalize the dataset
UOW_dataset_train_normalized <- apply(train[, -1], 2, normalize)

# after normalization, compute the range
after_range <- apply(UOW_dataset_train_normalized, 2, range)

# Range plot before normalization
plot(before_range, main = "Range Before Normalization", xlab = "Features", ylab = "Range")


#range of the plot after normalizing
plot(after_range, main = "Range After Normalization", xlab = "Features", ylab = "Range")

# All columns in the testing set should be normalized, with the exception of the date column.
UOW_dataset_test_normalized <- apply(test[, -1], 2, normalize)

# Rename columns in the testing set to correspond with the names of the columns in the training set.
colnames(UOW_dataset_test_normalized) <- colnames(UOW_dataset_train_normalized)


# Add the 18th and 19th attributes to the input vectors
# Give the input vectors a character vector list definition.
vectors_narx_input <- list(
  c("lag_1", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_7", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19")
)

# Build NARX models
# Create a list that is empty to hold the models.
models_narx <- list()
summary(UOW_dataset_train_normalized)
summary(UOW_dataset_test_normalized)
# Use a for loop to iterate over the input vectors
for (i in 1:length(narx_input_vectors)) {
  # Build a MLP model using the build_neural_net function, passing in the normalized training and test datasets,
  models_narx[[i]] <- build_neural_net(UOW_dataset_train_normalized, UOW_dataset_test_normalized,vectors_narx_input[[i]], c(5))
}

UOW_dataset_test_normalized <- as.data.frame(UOW_dataset_test_normalized)

# Evaluate NARX models
# Define an empty list to store the evaluation metrics
metrics_narx_evaluation <- list()
# Use a for loop to iterate over the models
for (i in 1:length(narx_models)) {
  # Calculate the evaluation metrics (RMSE, MAE, MAPE, and sMAPE) for each model using the calculate_metrics function,
  # passing in the actual test set values and the predictions from the current model
  metrics_narx_evaluation[[i]] <- metrics_calculation(UOW_dataset_test_normalized$hour_20, narx_models[[i]]$predictions)
}

# Create a comparison table for NARX models
# Create a data frame containing the Model_Description, RMSE, MAE, MAPE, and sMAPE columns
table_narx_comparison <- data.frame(
  Model_Description = c("NARX(1,18,19)", "NARX(2,18,19)", "NARX(3,18,19)", "NARX(3,7,18,19)", "NARX(4,7,18,19)"),
  RMSE = sapply(metrics_narx_evaluation, function(x) x$RMSE),
  MAE = sapply(metrics_narx_evaluation, function(x) x$MAE),
  MAPE = sapply(metrics_narx_evaluation, function(x) x$MAPE),
  sMAPE = sapply(metrics_narx_evaluation, function(x) x$sMAPE)
)
# Print the comparison table to the console
print(narx_comparison_table)


metrics_evaluation <- list()

for (i in 1:length(narx_models)) {
  metrics_evaluation[[i]] <- metrics_calculation(UOW_dataset_test_normalized$hour_20, narx_models[[i]]$predictions)
}

# Create 12–15 additional models with various hidden layer structures and input vectors.

# One-hidden-layer and two-hidden-layer networks are compared in terms of efficiency.

# Construct a single hidden layer neural network.
hidden_model_1 <- build_neural_net(UOW_dataset_train_normalized, UOW_dataset_test_normalized, c("lag_1", "hour_18", "hour_19"), c(5))

# Create a neural network with two hidden layers.
hidden_model_2 <- build_neural_net(UOW_dataset_train_normalized, UOW_dataset_test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19"), c(3, 2))

# The total amount of weight parameters per network is ridiculous
hidden_num_weights_1 <- sum(sapply(hidden_model_1$model$weights, length))
hidden_num_weights_2 <- sum(sapply(model_2_hidden$model$weights, length))

# For each network, print the number of weight parameters.
cat("Total number of weight parameters for the one-hidden layer network:", hidden_num_weights_1, "\n")
cat("Total number of weight parameters for the two-hidden layer network:", hidden_num_weights_2, "\n")


# Denormalize the predictions
denormalize <- function(x, min_value, max_value) {
  return(x * (max_value - min_value) + min_value)
}
# Based on the RMSE evaluation metric, identify the optimal model's index.
index_best_model <- which.min(sapply(evaluation_metrics, function(x) x$RMSE))

# Get the best model and its predictions
# check the length of models list
length(narx_models)

# change best_model_index to a legitimate index
index_best_model <- 1

# choose the best model
best_model <- narx_models[[best_model_index]]
predictions_best_model <- best_model$predictions

# Determine the variable 'hour_20's' minimum and maximum values in the training set.
min_value <- min(train$hour_20)
max_value <- max(train$hour_20)

# Denormalize the model predictions using the minimum and maximum values of the 'hour_20' variable.
predictions_denormalized <- denormalize(predictions_best_model, min_value, max_value)

# Use a line chart to compare the expected and desired results.
plot(test$hour_20, type = "l", col = "blue", xlab = "Time", ylab = "Hour 20 Consumption", main = "Line Chart of Desired vs. Predicted Output")
lines(predictions_denormalized, col = "red")
legend("topleft", legend = c("Desired Output", "Predicted Output"), col = c("blue", "red"),lty=1,cex=0.8)