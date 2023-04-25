library(readxl)
library(NbClust)
library(cluster) # Load the 'cluster' package

# Load the dataset
vehicles <- read_xlsx("vehicles.xlsx")

# Select only the first 18 attributes
vehicles <- vehicles[,1:18]
# Set smaller margins
par(mar = c(5, 5, 2, 2))

# Scale the data
scaled_data <- scale(vehicles)

# Detect and remove outliers using IQR method
Q1 <- apply(scaled_data, 2, quantile, probs = 0.25, na.rm = TRUE)
Q3 <- apply(scaled_data, 2, quantile, probs = 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
threshold <- 1.5 * IQR
outlier_indices <- which(apply(scaled_data, 2, function(x) any(x < (Q1 - threshold) | x > (Q3 + threshold))))
vehicles <- vehicles[, -outlier_indices]
boxplot(outlier_indices)
#-------------------------------------------------------------------------------------------------------------
#nb cluster
set.seed(123)# Set the random seed for reproducibility
par(mar=c(1,1,1,1))
nb <- NbClust(scaled_data, min.nc=2, max.nc=10, method="kmeans")
nb$Best.nc

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
                    K.max = 10, B = 50)
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


# Show the clustered results
cluster_results <- as.data.frame(km$cluster)
colnames(cluster_results) <- "Cluster"
print(cluster_results)

# Calculate the ratio of BSS over TSS
BSS <- km$betweenss
TSS <- km$totss
ratio_BSS_TSS <- BSS/TSS
print(ratio_BSS_TSS)

# Calculate the BSS and WSS indices
BSS_indices <- numeric(k)
WSS_indices <- numeric(k)
for (i in 1:k) {
  BSS_indices[i] <- sum((km$centers[i,] - mean(scaled_data))^2) * sum(km$size[km$cluster==i])
  WSS_indices[i] <- sum((scaled_data[km$cluster==i,] - km$centers[i,])^2)
}
print(BSS_indices)
print(WSS_indices)

#--------------------------------------------------------------------------------------------------------------

# Create a silhouette plot
sil <- silhouette(km$cluster, dist(scaled_data))
plot(sil)

# Calculate the average silhouette width score
sil_width <- summary(sil)$avg.width
cat("Average silhouette width score:", sil_width, "\n")

# Show the related kmeans output
print(km)

# Show the information for the centers
print(km$centers)

# Show the clustered results
cluster_results <- as.data.frame(km$cluster)
colnames(cluster_results) <- "Cluster"
print(cluster_results)

#--------------------------------------------------------------------------------------------------------------


