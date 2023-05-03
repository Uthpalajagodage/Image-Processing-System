#Import library
library(readxl)
library(neuralnet)
library(ggplot2)
library(MLmetrics)
library(keras)

#PART 1
# Define the root-mean-square error (RMSE) function
rmse <- function(error) {
  return(sqrt(mean(error^2)))
}

# Define the mean absolute error (MAE) function
mae <- function(error) {
  return(mean(abs(error)))
}

# Define the mean absolute percentage error (MAPE) function
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Define the symmetric mean absolute percentage error (sMAPE) function
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted)))*100)
}

# Load the UOW consumption dataset
UOW_consumption_data <- read_xlsx("datasets/uow_consumption.xlsx")



summary(UOW_consumption_data) # Get summary statistics for the column


# Extract the hourly electricity consumption data for 20:00 for 2018 and 2019
#   "0.83333333333333337"= 20.00  (20/24 = 0.83333333333333337)
consumption_hourly_20 <- UOW_consumption_data[c("date", "0.83333333333333337")]

# plot the hourly consumption data for 20:00 for 2018 and 2019
ggplot(UOW_consumption_data, aes(x=date, y=`0.83333333333333337`)) +
  geom_line() +
  labs(title = "Hourly Consumption for 20:00",
       x = "Date",
       y = "Consumption")


# Extract the first 380 samples as training data, and the remaining samples as testing data
trained_data <- unlist(consumption_hourly_20[1:380, "0.83333333333333337"])
test_dataset <- unlist(consumption_hourly_20[381:nrow(consumption_hourly_20), "0.83333333333333337"])

# plot the first 380 samples as training data
ggplot(data.frame(trained_data), aes(x=1:length(trained_data), y=trained_data)) +
  geom_line() +
  labs(title = "Training Data",
       x = "Sample Number",
       y = "Consumption")


# Define the number of time-delayed inputs
no_inputs <- 60


# Construct the input/output matrix for MLP training/testing
matrix_input_output <- matrix(0, nrow=length(trained_data)-no_inputs, ncol=no_inputs+1)

for (i in 1:(length(trained_data)-no_inputs)) {
  matrix_input_output[i, 1:no_inputs] <- trained_data[i:(i+no_inputs-1)]
  matrix_input_output[i, no_inputs+1] <- trained_data[i+no_inputs]
}

# Normalize the input/output matrix
matrix_input_output <- apply(matrix_input_output, 2, function(x) (x - mean(x)) / sd(x))

# Define the neural network structure to be evaluated
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
  
  # Train the MLP using the normalized input/output matrix
  mlp <- neuralnet(V2 ~ ., data=matrix_input_output, hidden=structure[[i]], linear.output=TRUE)
  
  # #Plot the neural network
  # plot(mlp)
  
  # Extract the inputs for the test data
  testing_inputs <- matrix(test_dataset[1:(length(test_dataset)-no_inputs)], ncol=no_inputs, byrow=TRUE)
  
  # Predict the output values for the test data
output_mlp <- predict(mlp, testing_inputs)
  
  # Denormalize the predicted output values
output_mlp <- (mlp_output * sd(trained_data)) + mean(trained_data)
  
  # Calculate the MAE for the predicted output values and the actual output values
  result_mae <- mae(mlp_output - test_dataset[(no_inputs+1):length(test_dataset)])
  
  cat("the test performances for c(",structure[[i]],")\n")
  
  # Print the MAE result
  cat("The MAE for the test data is:", round(result_mae, 2),"\n")
  
  # Calculate the RMSE for the predicted output values and the actual output values
result_rmse <- rmse(mlp_output - test_dataset[(no_inputs+1):length(test_dataset)])
  
  # Print the RMSE result
  cat("The RMSE for the test data is:", round(result_rmse, 2),"\n")
  
  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }
  
  # Calculate the MAPE for the predicted output values and the actual output values
result_mape <- mape(test_dataset[(no_inputs+1):length(test_dataset)], output_mlp)
  
  # Print the MAPE result
  cat("The MAPE for the test data is:", round(result_mape, 2),"\n")
  
  # Define the symmetric mean absolute percentage error (sMAPE) function
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }
  
  # Calculate the sMAPE for the predicted output values and the actual output values
  result_smape <- smape(test_dataset[(no_inputs+1):length(test_dataset)], mlp_output)
  
  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(result_smape, 2),"\n\n")
  
  # Store the result for the current neural network structure
  result[[i]] <- c(structure[[i]], result_mae, result_rmse, result_mape, result_smape)
  
}

# Create a data frame of the result
results_dframe <- data.frame(matrix(unlist(result), ncol=5, byrow=TRUE))
colnames(results_dframe) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_dframe)

# Find the best one-hidden and two-hidden layer structure based on MAE and total number of weights
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


# Define a function to build a neural network model
build_neural_network <- function(trained_data, test_dataset, input_vars, hidden_structure) {
  
  # Create formula for the neural network
  formula <- paste("hour_20 ~", paste(input_vars, collapse = " + "))
  
  # Build the neural network model using the neuralnet package
 model_nn <- neuralnetwork(as.formula(formula), trained_data, hidden = hidden_structure)
  
  # Prepare the test data for prediction
matrix_test <- as.matrix(test_dataset[, input_vars, drop = FALSE])
  colnames(matrix_test) <- colnames(trained_data[, input_vars, drop = FALSE])
  
  # Make predictions using the neural network model
  predictions <- predict(model_nn, matrix_test)
  
  # Return the neural network model and its predictions
  return(list(model = model_nn, predictions = predictions))
}

# Function to calculate different evaluation metrics
metrics_calculation <- function(actual_val, predicted_val) {
  # Calculate Root Mean Squared Error
  rmse <- sqrt(mean((actual_val - predicted_val)^2))
  
  # Calculate Mean Absolute Error
  mae <- mean(abs(actual_val - predicted_val))
  
  # Calculate Mean Absolute Percentage Error
  mape <- mean(abs((actual_val - predicted_val) / actual_val)) * 100
  
  # Calculate Symmetric Mean Absolute Percentage Error
  smape <- mean(abs(actual_val - predicted_val) / (abs(actual_val) + abs(predicted_val)) * 2) * 100
  
  # Return a list containing all the evaluation metrics
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}


# Rename columns to be more descriptive
colnames(UOW_consumption_data) <- c("date", "hour_18", "hour_19", "hour_20")

# Create lagged variables for hour_20, with different time lags
UOW_consumption_data$lag_1 <- lag(UOW_consumption_data$hour_20, 1)
UOW_consumption_data$lag_2 <- lag(UOW_consumption_data$hour_20, 2)
UOW_consumption_data$lag_3 <- lag(UOW_consumption_data$hour_20, 3)
UOW_consumption_data$lag_4 <- lag(UOW_consumption_data$hour_20, 4)
UOW_consumption_data$lag_7 <- lag(UOW_consumption_data$hour_20, 7)

# Remove rows with missing values
UOW_consumption_data <- na.omit(UOW_consumption_data)

# Split data into training and testing sets based on row index
train <- UOW_consumption_data[1:380,]
test <- UOW_consumption_data[381:nrow(UOW_consumption_data),]

# Define normalization function to scale data between 0 and 1
normalized <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Compute range before normalization
before_range <- apply(train[, -1], 2, range)


# Normalize the dataset
UOW_dataset_train_normalized <- apply(train[, -1], 2, normalize)

# Compute range after normalization
after_range <- apply(UOW_dataset_train_normalized, 2, range)

# Plot range before normalization
plot(before_range, main = "Range Before Normalization", xlab = "Features", ylab = "Range")


# Plot range after normalization
plot(after_range, main = "Range After Normalization", xlab = "Features", ylab = "Range")



# Apply normalization function to all columns except the date column in the testing set
UOW_dataset_test_normalized <- apply(test[, -1], 2, normalize)

# Rename columns in the testing set to match the column names in the training set
colnames(UOW_dataset_test_normalized) <- colnames(UOW_dataset_train_normalized)


# Add the 18th and 19th hour attributes to the input vectors
# Define the input vectors as a list of character vectors
vectors_narx_input <- list(
  c("lag_1", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_7", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19")
)

# Build NARX models
# Define an empty list to store the models
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

# Add more models with different hidden layer structure and input vectors to create 12-15 models in total

# Efficiency comparison between one-hidden layer and two-hidden layer networks

# Build a one-hidden layer neural network
hidden_model_1 <- build_neural_net(UOW_dataset_train_normalized, UOW_dataset_test_normalized, c("lag_1", "hour_18", "hour_19"), c(5))

# Build a two-hidden layer neural network
hidden_model_2 <- build_neural_net(UOW_dataset_train_normalized, UOW_dataset_test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19"), c(3, 2))

# Check the total number of weight parameters per network
hidden_num_weights_1 <- sum(sapply(hidden_model_1$model$weights, length))
hidden_num_weights_2 <- sum(sapply(model_2_hidden$model$weights, length))

# Print the number of weight parameters for each network
cat("Total number of weight parameters for the one-hidden layer network:", hidden_num_weights_1, "\n")
cat("Total number of weight parameters for the two-hidden layer network:", hidden_num_weights_2, "\n")


# Denormalize the predictions
denormalize <- function(x, min_value, max_value) {
  return(x * (max_value - min_value) + min_value)
}


# Find the index of the best model based on the RMSE evaluation metric
index_best_model <- which.min(sapply(evaluation_metrics, function(x) x$RMSE))

# Get the best model and its predictions
# check the length of models list
length(narx_models)

# set best_model_index to a valid index
index_best_model <- 1

# get the best model
best_model <- narx_models[[best_model_index]]
predictions_best_model <- best_model$predictions

# Find the minimum and maximum values of the 'hour_20' variable in the training set
min_value <- min(train$hour_20)
max_value <- max(train$hour_20)

# Denormalize the model predictions using the min and max values of the 'hour_20' variable
predictions_denormalized <- denormalize(predictions_best_model, min_value, max_value)

# Plot the predicted output vs. desired output using a line chart
plot(test$hour_20, type = "l", col = "blue", xlab = "Time", ylab = "Hour 20 Consumption", main = "Line Chart of Desired vs. Predicted Output")
lines(predictions_denormalized, col = "red")
legend("topleft", legend = c("Desired Output", "Predicted Output"), col = c("blue", "red"),lty=1,cex=0.8)