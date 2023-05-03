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
uow_consumption_dataset <- read_xlsx("datasets/uow_consumption.xlsx")



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