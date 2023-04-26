library(readxl)

# Determine the optimal number of clusters using NbClust
library(NbClust)

# Elbow Method
library(ggplot2)

# Load required packages
library(cluster)
library(factoextra)


# Load the dataset
vehicles <- read_xlsx("vehicles.xlsx")

# Select only the first 18 attributes
vehicles <- vehicles[,1:18]
# Set smaller margins
par(mar = c(5, 5, 2, 2))

# Scale the data
scaled_data <- scale(vehicles)


# Detect and remove outliers using the Z-score method
z_scores <- apply(scaled_data, 1, function(x) sum(abs(x) > 3))
outliers <- which(z_scores > 0)
scaled_data <- scaled_data[-outliers,]

#-------------------------------------------------------------------------------------------------------------
#nb cluster
# Perform k-means clustering on the pre-processed data
set.seed(123)
par(mar=c(1,1,1,1))
nbclust_index <- NbClust(scaled_data, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")


#--------------------------------------------------------------------------------------------------------------

#Elbow methods
wcss <- vector("numeric", length = 5)
for (i in 2:5) {
  kmeans_model <- kmeans(scaled_data, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}

# Plot the elbow curve
plot(1:5, wcss, type = "b", xlab = "Number of clusters", ylab = "WCSS")
title(main = "Elbow curve for k-means clustering")
abline(v = 3, col = "red", lty = 2)

# Find the "elbow" in the plot
diffs <- diff(wcss)
elbow <- which(diffs == min(diffs)) + 1

# Print the best number of clusters based on the elbow method
cat("Best number of clusters based on the elbow method:",elbow,"\n")
#--------------------------------------------------------------------------------------------------------------

#Gap statistics
gap_stat <- clusGap(scaled_data, FUN = kmeans, nstart = 25,
                    K.max = 3, B = 50)
plot(gap_stat, main = "Gap Statistic plot for Vehicle Dataset")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k,"\n")

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

# Check if elbow was able to find an optimal number of clusters
if (!(elbow > 1)) {
  cat("Error: elbow could not find an optimal number of clusters. Try adjusting the parameters or preprocessing steps.")
} else {
  # Perform k-means clustering on the pre-processed data
  set.seed(123)
  par(mar=c(1,1,1,1))
  k <- elbow
  kmeans_fit <- kmeans(scaled_data, centers = k, nstart = 25)
  
  # Print k-means output
  cat("K-means clustering with", k, "clusters\n")
  print(kmeans_fit$centers)
  print(kmeans_fit$cluster)
  
  # Calculate BSS and WSS
  TSS <- sum(apply(scaled_data, 2, var))
  BSS <- sum(kmeans_fit$size * apply(kmeans_fit$centers, 2, var))
  WSS <- sum(kmeans_fit$withinss)
  
  # Print BSS/TSS ratio and WSS/BSS ratio
  cat("BSS/TSS ratio:", BSS/TSS, "\n")
  cat("WSS/BSS ratio:", WSS/BSS, "\n")
  cat("TSS_indices : ",TSS, "\n")
  cat("BSS_indices : ",BSS, "\n")
  cat("WSS_indices : ",WSS,"\n")
  
}

#--------------------------------------------------------------------------------------------------------------

# Plot the clustering results
par(mar=c(1,1,1,1))
plot(scaled_data, col = kmeans_result$cluster)
points(kmeans_result$centers, col = 1:kmeans_result$cluster, pch = 8, cex = 2)

# Calculate silhouette coefficients and plot the silhouette plot
silhouette_obj <- silhouette(kmeans_result$cluster, dist(scaled_data))
plot(silhouette_obj)

# Calculate the average silhouette width score
avg_sil_width <- mean(silhouette_obj[, 3])
cat("Average Silhouette Width Score:", avg_sil_width,"\n")
#--------------------------------------------------------------------------------------------------------------


