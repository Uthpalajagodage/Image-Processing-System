
library(readxl)
library(NbClust)
library(cluster) # Load the 'cluster' package, which contains the kmeans and silhouette functions



vehicles <- read_excel("D:/GIT/Image-Processing-System/MachineLearning/datasets/vehicles.xlsx")
print(vehicles)

# Select only the first 18 attributes
vehicles <- vehicles[,1:18]
plot(vehicles)



# Scale the data
scaled_data <- scale(vehicles)
plot(scaled_data)

# Detect and remove outliers using IQR method
Q1 <- apply(scaled_data, 2, quantile, probs = 0.25, na.rm = TRUE)
Q3 <- apply(scaled_data, 2, quantile, probs = 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
threshold <- 1.5 * IQR
outlier_indices <- which(apply(scaled_data, 2, function(x) any(x < (Q1 - threshold) | x > (Q3 + threshold))))
vehicles <- vehicles[, -outlier_indices]


#nb cluster
set.seed(123)
nb <- NbClust(scaled_data, min.nc=2, max.nc=10, method="kmeans")
nb$Best.nc

#set.seed(123)
wss <- c()
for(i in 2:10) wss[i] <- sum(kmeans(scaled_data, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Number of clusters", ylab="Within groups sum of squares")


#set.seed(123)
gap_stat <- clusGap(scaled_data, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
plot(gap_stat, main = "Gap Statistic plot for Vehicle Dataset")

cat("Best number of clusters based on the gap statistic method:", gap_stat$Tab)



# Set the random seed for reproducibility
#set.seed(123)

#--------------------------------------------------------------------------------------------------------------
# Calculate the average silhouette width for different values of k
# Set the range of K values to test
k.min <- 2
k.max <- 10

# Create a list to store the silhouette values for each value of K
silhouette_vals <- vector("list", k.max - k.min + 1)

# Loop through each value of K and perform clustering using K-means algorithm
for (k in k.min:k.max) {
  km <- kmeans(scaled_data, centers = k, nstart = 10)
  
  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(scaled_data))
}

# Calculate the average silhouette width for each value of K
silhouette_avg <- sapply(silhouette_vals, function(x) mean(x[, 3]))

# Plot the silhouette widths for each value of K
plot(k.min:k.max, silhouette_avg, type = "b", xlab = "Number of clusters", ylab="Silhouette")

# Find the index of the maximum silhouette width
best_k <- which.max(silhouette_avg) + k.min - 1

# Print the best number of clusters based on the silhouette method
cat("Best number of clusters based on the silhouette method: ", best_k, "\n")


#--------------------------------------------------------------------------------------------------------------

