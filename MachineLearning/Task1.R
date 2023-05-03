#Task 1

library(readxl)

# Determine the optimal number of clusters using NbClust
library(NbClust)

# Elbow Method
library(ggplot2)

# Load required packages
library(cluster)
library(factoextra)

# Load the required package for PCA
library(FactoMineR)

library(fpc)



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
# nb cluster
# Perform k-means clustering on the pre-processed data
set.seed(123)
par(mar=c(1,1,1,1))
nbclust_index <- NbClust(scaled_data, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Reshape NbClust results to a long format
df_long <- data.frame(
  Clusters = rep(2:10, each = ncol(nbclust_index$All.index)),
  Index = rep(colnames(nbclust_index$All.index), times = 9),
  Value = as.vector(nbclust_index$All.index)
)

# Plot the bar plot using ggplot2
ggplot(df_long, aes(x = Clusters, y = Value, fill = Index)) +
  geom_bar(stat = "identity", position = "dodge") +
  xlab("Number of clusters") +
  ylab("Clustering index") +
  ggtitle("NbClust plot") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_vline(xintercept = nbclust_index$Best.nc[1], linetype = "dashed",color="blue")

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
plot(scaled_data, col = kmeans_fit$cluster)
points(kmeans_fit$centers, col = 1:kmeans_fit$cluster, pch = 8, cex = 2)

# Calculate silhouette coefficients and plot the silhouette plot
silhouette_obj <- silhouette(kmeans_fit$cluster, dist(scaled_data))
plot(silhouette_obj)

# Calculate the average silhouette width score
avg_sil_width <- mean(silhouette_obj[, 3])
cat("Average Silhouette Width Score:", avg_sil_width,"\n")
--------------------------------------------------------------------------------------------------------------

# Separate the class label from the data
vehicle_class <- scaled_data[,1]
vehicles <- scaled_data[,-1]

# Perform PCA analysis on the data
pca_result <- PCA(vehicles, graph=FALSE)

# Show the eigenvalues of the principal components
print(pca_result$eig)

# Show the cumulative score per principal component
print(pca_result$eig[2,])

# Create a new dataset with principal components as attributes
num_pcs <- sum(pca_result$eig[2,] <= 0.92)
transformed_data <- pca_result$ind$coord[,1:num_pcs]
print(transformed_data)

#--------------------------------------------------------------------------------------------------------------

# nb cluster
# Perform k-means clustering on the pre-processed data
set.seed(123)
par(mar=c(1,1,1,1))
nbclust_index <- NbClust(transformed_data, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Reshape NbClust results to a long format
df_long <- data.frame(
  Clusters = rep(2:10, each = ncol(nbclust_index$All.index)),
  Index = rep(colnames(nbclust_index$All.index), times = 9),
  Value = as.vector(nbclust_index$All.index)
)

# Plot the bar plot using ggplot2
ggplot(df_long, aes(x = Clusters, y = Value, fill = Index)) +
  geom_bar(stat = "identity", position = "dodge") +
  xlab("Number of clusters") +
  ylab("Clustering index") +
  ggtitle("NbClust plot") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_vline(xintercept = nbclust_index$Best.nc[1], linetype = "dashed",color="blue")
#--------------------------------------------------------------------------------------------------------------

#Elbow methods
wcss <- vector("numeric", length = 5)
for (i in 2:5) {
  kmeans_model <- kmeans(transformed_data, centers = i, nstart = 25)
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
gap_stat <- clusGap(as.matrix(transformed_data), FUN = kmeans, nstart = 25,
                     K.max = 3, B = 50)

plot(gap_stat, main = "Gap Statistic plot for Vehicle Dataset")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k,"\n")
#--------------------------------------------------------------------------------------------------------------

# Calculate the average silhouette width for different values of k
# Set the range of K values to test

# Create a list to store the silhouette values for each value of K
silhouette_vals <- vector("list", k.max - k.min + 1)

# Loop through each value of K and perform clustering using K-means algorithm
k.min <- 2
k.max <- 5
silhouette_vals <- vector("list", length = k.max - k.min + 1)

for (k in k.min:k.max) {
  km <- kmeans(transformed_data, centers = k, nstart = 10)

  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(transformed_data))
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

# # Check if elbow was able to find an optimal number of clusters
if (!(elbow > 1)) {
  cat("Error: elbow could not find an optimal number of clusters. Try adjusting the parameters or preprocessing steps.")
} else {
#Perform k-means clustering on the pre-processed data
# Reshape the vector "transformed_vehicles_data" into a matrix with a single column
transformed_data <- matrix(transformed_data,ncol=1)
  set.seed(123)
  par(mar=c(1,1,1,1))
  k <- elbow
  # Perform k-means clustering on the transformed data with the best k value
  set.seed(123)
  kmeans_transf <- kmeans(transformed_data, k)

  # Plot the clustering results using elbow method
  plot(transformed_data, col = kmeans_transf$cluster)
  points(kmeans_transf$centers, col = 1:k, pch = 8, cex = 2)

  # Print the k-means output
  print(kmeans_transf)

  # Calculate BSS and WSS for transformed_vehicles_data
  BSS_transf <- sum(kmeans_transf$size * apply((kmeans_transf$centers - apply(transformed_data, 2, mean))^2, 1, sum))
  TSS_transf <- sum(apply(transformed_data^2, 1, sum)) - (sum(transformed_data)^2)/length(transformed_data)
  WSS_transf <- kmeans_transf$tot.withinss

  # Print BSS/TSS ratio, BSS, and WSS for transformed_vehicles_data
  cat("BSS/TSS ratio for transformed_vehicles_data: ", BSS_transf/TSS_transf, "\n")
  cat("TSS for transformed_vehicles_data: ", TSS_transf, "\n")
  cat("BSS for transformed_vehicles_data: ", BSS_transf, "\n")
  cat("WSS for transformed_vehicles_data: ", WSS_transf, "\n")
  }

#--------------------------------------------------------------------------------------------------------------
# Plot the clustering results

par(mar=c(1,1,1,1))
plot(transformed_data, col = kmeans_transf$cluster)
points(kmeans_transf$centers, col = 1:kmeans_fit$cluster, pch = 8, cex = 2)

# Calculate silhouette coefficients and plot the silhouette plot
silhouette_obj <- silhouette(kmeans_transf$cluster, dist(transformed_data))
plot(silhouette_obj)

# Calculate the average silhouette width score
avg_sil_width <- mean(silhouette_obj[, 3])
cat("Average Silhouette Width Score:", avg_sil_width,"\n")
#--------------------------------------------------------------------------------------------------------------
# Calculation of the Calinski-Harabasz index for evaluating clustering performance
calinski_harabasz_pca <- function(cluster_result, data) {
  k <- length(unique(cluster_result$cluster))
  n <- nrow(data)
  BSS <- cluster_result$betweenss
  WSS <- cluster_result$tot.withinss
  
  ch_index <- ((n - k) / (k - 1)) * (BSS / WSS)
  return(ch_index)
}

ch_index_pca <- calinski_harabasz_pca(kmeans_transf, transformed_data)
ch_index_pca