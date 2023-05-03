#Import library
library(readxl)
library(neuralnet)
library(ggplot2)
library(MLmetrics)
library(keras)


#Part 1

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
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
}


# Load the UOW consumption dataset
uow_consumption_dataset <- read_xlsx("data sets/uow_consumption.xlsx")

summary(uow_consumption_dataset) # Get summary statistics for the column


# Extract the hourly electricity consumption data for 20:00 for 2018 and 2019
#   "0.83333333333333337"= 20.00  (20/24 = 0.83333333333333337)
hourly_consumption_20 <- uow_consumption_dataset[c("date", "0.83333333333333337")]

# plot the hourly consumption data for 20:00 for 2018 and 2019
ggplot(uow_consumption_dataset, aes(x=date, y=`0.83333333333333337`)) +
  geom_line() +
  labs(title = "Hourly Consumption for 20:00",
       x = "Date",
       y = "Consumption")


# Extract the first 380 samples as training data, and the remaining samples as testing data
train_data <- unlist(hourly_consumption_20[1:380, "0.83333333333333337"])
test_data <- unlist(hourly_consumption_20[381:nrow(hourly_consumption_20), "0.83333333333333337"])

# plot the first 380 samples as training data
ggplot(data.frame(train_data), aes(x=1:length(train_data), y=train_data)) +
  geom_line() +
  labs(title = "Training Data",
       x = "Sample Number",
       y = "Consumption")


# Define the number of time-delayed inputs
num_inputs <- 60


# Construct the input/output matrix for MLP training/testing
input_output_matrix <- matrix(0, nrow=length(train_data)-num_inputs, ncol=num_inputs+1)

for (i in 1:(length(train_data)-num_inputs)) {
  input_output_matrix[i, 1:num_inputs] <- train_data[i:(i+num_inputs-1)]
  input_output_matrix[i, num_inputs+1] <- train_data[i+num_inputs]
}

# Normalize the input/output matrix
input_output_matrix <- apply(input_output_matrix, 2, function(x) (x - mean(x)) / sd(x))

# Define the neural network structures to be evaluated
structures <- list(
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

results <- list()

for (i in 1:length(structures)) {
  
  # Train the MLP using the normalized input/output matrix
  mlp <- neuralnet(V2 ~ ., data=input_output_matrix, hidden=structures[[i]], linear.output=TRUE)
  
  # #Plot the neural network
  # plot(mlp)
  
  # Extract the inputs for the test data
  test_inputs <- matrix(test_data[1:(length(test_data)-num_inputs)], ncol=num_inputs, byrow=TRUE)
  
  # Predict the output values for the test data
  mlp_output <- predict(mlp, test_inputs)
  
  # Denormalize the predicted output values
  mlp_output <- (mlp_output * sd(train_data)) + mean(train_data)
  
  # Calculate the MAE for the predicted output values and the actual output values
  mae_result <- mae(mlp_output - test_data[(num_inputs+1):length(test_data)])
  
  cat("the test performances for c(",structures[[i]],")\n")
  
  # Print the MAE result
  cat("The MAE for the test data is:", round(mae_result, 2),"\n")
  
  # Calculate the RMSE for the predicted output values and the actual output values
  rmse_result <- rmse(mlp_output - test_data[(num_inputs+1):length(test_data)])
  
  # Print the RMSE result
  cat("The RMSE for the test data is:", round(rmse_result, 2),"\n")
  
  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }
  
  # Calculate the MAPE for the predicted output values and the actual output values
  mape_result <- mape(test_data[(num_inputs+1):length(test_data)], mlp_output)
  
  # Print the MAPE result
  cat("The MAPE for the test data is:", round(mape_result, 2),"\n")
  
  # Define the symmetric mean absolute percentage error (sMAPE) function
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }
  
  # Calculate the sMAPE for the predicted output values and the actual output values
  smape_result <- smape(test_data[(num_inputs+1):length(test_data)], mlp_output)
  
  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(smape_result, 2),"\n\n")
  
  # Store the results for the current neural network structure
  results[[i]] <- c(structures[[i]], mae_result, rmse_result, mape_result, smape_result)
  
}

# Create a data frame of the results
results_df <- data.frame(matrix(unlist(results), ncol=5, byrow=TRUE))
colnames(results_df) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_df)

# Find the best one-hidden and two-hidden layer structures based on MAE and total number of weights
best_one_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]
best_two_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]

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
build_neural_net <- function(train_data, test_data, input_vars, hidden_structure) {
  
  # Create formula for the neural network
  formula <- paste("hour_20 ~", paste(input_vars, collapse = " + "))
  
  # Build the neural network model using the neuralnet package
  nn_model <- neuralnet(as.formula(formula), train_data, hidden = hidden_structure)
  
  # Prepare the test data for prediction
  test_matrix <- as.matrix(test_data[, input_vars, drop = FALSE])
  colnames(test_matrix) <- colnames(train_data[, input_vars, drop = FALSE])
  
  # Make predictions using the neural network model
  predictions <- predict(nn_model, test_matrix)
  
  # Return the neural network model and its predictions
  return(list(model = nn_model, predictions = predictions))
}

# Function to calculate different evaluation metrics
calculate_metrics <- function(actual_values, predicted_values) {
  # Calculate Root Mean Squared Error
  rmse <- sqrt(mean((actual_values - predicted_values)^2))
  
  # Calculate Mean Absolute Error
  mae <- mean(abs(actual_values - predicted_values))
  
  # Calculate Mean Absolute Percentage Error
  mape <- mean(abs((actual_values - predicted_values) / actual_values)) * 100
  
  # Calculate Symmetric Mean Absolute Percentage Error
  smape <- mean(abs(actual_values - predicted_values) / (abs(actual_values) + abs(predicted_values)) * 2) * 100
  
  # Return a list containing all the evaluation metrics
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}


# Rename columns to be more descriptive
colnames(uow_consumption_dataset) <- c("date", "hour_18", "hour_19", "hour_20")

# Create lagged variables for hour_20, with different time lags
uow_consumption_dataset$lag_1 <- lag(uow_consumption_dataset$hour_20, 1)
uow_consumption_dataset$lag_2 <- lag(uow_consumption_dataset$hour_20, 2)
uow_consumption_dataset$lag_3 <- lag(uow_consumption_dataset$hour_20, 3)
uow_consumption_dataset$lag_4 <- lag(uow_consumption_dataset$hour_20, 4)
uow_consumption_dataset$lag_7 <- lag(uow_consumption_dataset$hour_20, 7)

# Remove rows with missing values
uow_consumption_dataset <- na.omit(uow_consumption_dataset)

# Split data into training and testing sets based on row index
train <- uow_consumption_dataset[1:380,]
test <- uow_consumption_dataset[381:nrow(uow_consumption_dataset),]

# Define normalization function to scale data between 0 and 1
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}



# Compute the range of each column before and after normalization
train_range <- apply(train[, -1], 2, range)
train_normalized_range <- apply(uow_dataset_train_normalized[, -1], 2, range)

# Create a matrix with the range values
range_matrix <- rbind(train_range[1, ], train_normalized_range[1, ])

# Define the x-axis labels
x_labels <- colnames(train)[-1]

# Plot the range values as a bar chart
barplot(range_matrix, beside = TRUE, col = c("red", "blue"),
        xlab = "Column", ylab = "Value", main = "Effect of Normalization on Value Range",
        names.arg = x_labels, legend.text = c("Before Normalization", "After Normalization"))


# Apply normalization function to all columns except the date column in the testing set
uow_dataset_test_normalized <- test
uow_dataset_test_normalized[, -1] <- apply(test[, -1], 2, normalize)

# Rename columns in the testing set to match the column names in the training set
colnames(uow_dataset_test_normalized) <- colnames(uow_dataset_train_normalized)


# Add the 18th and 19th hour attributes to the input vectors
# Define the input vectors as a list of character vectors
narx_input_vectors <- list(
  c("lag_1", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_7", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19")
)

# Build NARX models
# Define an empty list to store the models
narx_models <- list()
# Use a for loop to iterate over the input vectors
for (i in 1:length(narx_input_vectors)) {
  # Build a MLP model using the build_neural_net function, passing in the normalized training and test datasets,
  narx_models[[i]] <- build_neural_net(uow_dataset_train_normalized, uow_dataset_test_normalized, narx_input_vectors[[i]], c(5))
}

# Evaluate NARX models
# Define an empty list to store the evaluation metrics
narx_evaluation_metrics <- list()
# Use a for loop to iterate over the models
for (i in 1:length(narx_models)) {
  # Calculate the evaluation metrics (RMSE, MAE, MAPE, and sMAPE) for each model using the calculate_metrics function,
  # passing in the actual test set values and the predictions from the current model
  narx_evaluation_metrics[[i]] <- calculate_metrics(uow_dataset_test_normalized$hour_20, narx_models[[i]]$predictions)
}

# Create a comparison table for NARX models
# Create a data frame containing the Model_Description, RMSE, MAE, MAPE, and sMAPE columns
narx_comparison_table <- data.frame(
  Model_Description = c("NARX(1,18,19)", "NARX(2,18,19)", "NARX(3,18,19)", "NARX(3,7,18,19)", "NARX(4,7,18,19)"),
  RMSE = sapply(narx_evaluation_metrics, function(x) x$RMSE),
  MAE = sapply(narx_evaluation_metrics, function(x) x$MAE),
  MAPE = sapply(narx_evaluation_metrics, function(x) x$MAPE),
  sMAPE = sapply(narx_evaluation_metrics, function(x) x$sMAPE)
)
# Print the comparison table to the console
print(narx_comparison_table)


evaluation_metrics <- list()

for (i in 1:length(narx_models)) {
  evaluation_metrics[[i]] <- calculate_metrics(uow_dataset_test_normalized$hour_20, narx_models[[i]]$predictions)
}

# Add more models with different hidden layer structures and input vectors to create 12-15 models in total

# Efficiency comparison between one-hidden layer and two-hidden layer networks

# Build a one-hidden layer neural network
model_1_hidden <- build_neural_net(uow_dataset_train_normalized, uow_dataset_test_normalized, c("lag_1", "hour_18", "hour_19"), c(5))

# Build a two-hidden layer neural network
model_2_hidden <- build_neural_net(uow_dataset_train_normalized, uow_dataset_test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19"), c(3, 2))

# Check the total number of weight parameters per network
num_weights_1_hidden <- sum(sapply(model_1_hidden$model$weights, length))
num_weights_2_hidden <- sum(sapply(model_2_hidden$model$weights, length))

# Print the number of weight parameters for each network
cat("Total number of weight parameters for the one-hidden layer network:", num_weights_1_hidden, "\n")
cat("Total number of weight parameters for the two-hidden layer network:", num_weights_2_hidden, "\n")


# Denormalize the predictions
denormalize <- function(x, min_value, max_value) {
  return(x * (max_value - min_value) + min_value)
}


# Find the index of the best model based on the RMSE evaluation metric
best_model_index <- which.min(sapply(evaluation_metrics, function(x) x$RMSE))

# Get the best model and its predictions
# check the length of models list
length(narx_models)

# set best_model_index to a valid index
best_model_index <- 1

# get the best model
best_model <- narx_models[[best_model_index]]
best_model_predictions <- best_model$predictions

# Find the minimum and maximum values of the 'hour_20' variable in the training set
min_value <- min(train$hour_20)
max_value <- max(train$hour_20)

# Denormalize the model predictions using the min and max values of the 'hour_20' variable
denormalized_predictions <- denormalize(best_model_predictions, min_value, max_value)

# Plot the predicted output vs. desired output using a line chart
plot(test$hour_20, type = "l", col = "blue", xlab = "Time", ylab = "Hour 20 Consumption", main = "Line Chart of Desired vs. Predicted Output")
lines(denormalized_predictions, col = "red")
legend("topleft", legend = c("Desired Output", "Predicted Output"), col = c("blue", "red"), lty=1, cex=0.8)