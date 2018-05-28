library(mclust)
library(plotly)
library(ggplot)
library(ggplot2)
library(ggbiplot)
library(neuralnet)
library(caret)
library(fastICA)
library(mclust)
library(dplyr)



cardio <- read.csv("Cardiotocographic.csv")
cardio1 <- cardio[-c(22)]

# Create matrix
set.seed(123)
random <- runif(210, 0.000, 100.000)
m <- matrix(random, 21, 10)

# Random Projecation
cardio2 <- as.matrix(cardio1)
system.time(cardio3 <- cardio2 %*% m)
cardio4 <- as.data.frame(cardio3)

# Normalize
maxs <- apply(cardio4, 2, max)
mins <- apply(cardio4, 2, min)
scaled <- as.data.frame(scale(cardio4, center = mins, scale = maxs - mins))
head(scaled)
cardio_scaled <- cbind(scaled, cardio[c(22)])

# Binarize the categorical output
cardio_scaled <- cbind(cardio_scaled, cardio$NSP == 1)
cardio_scaled <- cbind(cardio_scaled, cardio$NSP == 2)
cardio_scaled <- cbind(cardio_scaled, cardio$NSP == 3)

names(cardio_scaled)[12] <- "normal"
names(cardio_scaled)[13] <- "suspect"
names(cardio_scaled)[14] <- "pathologic"

cols <- sapply(cardio_scaled, is.logical)
cardio_scaled[,cols] <- lapply(cardio_scaled[,cols], as.numeric)

df <- cardio_scaled[c(1:10)]

### K-means

# Scree Plot
wss <- (nrow(df))*sum(apply(df,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters(Cardio 3 Clusters)", ylab="Within groups sum of squares") 

# K-means clustering
kc <- kmeans(df, 3)

kc$cluster <- as.factor(kc$cluster)
plot_ly(x= df$V1, y= df$V2, color = kc$cluster)


### Neural Network

# Split data
sample2 <- sample(1:nrow(cardio_scaled), 0.7*nrow(cardio_scaled), replace = FALSE)
cardio_train <- cardio_scaled[sample2, ]
cardio_test <- cardio_scaled[-sample2, ]

# 
n <- names(cardio_train)
f <- as.formula(paste("normal + suspect + pathologic ~", 
                      paste(n[!n %in% c("normal", "suspect", "pathologic", "NSP")], 
                            collapse = " + ")))

set.seed(101)
system.time(
        cardio_nn_1 <- neuralnet(f,
                                 data = cardio_train, 
                                 hidden = c(5),
                                 threshold = 0.5,
                                 stepmax = 1e7,
                                 err.fct = "sse",
                                 act.fct = "tanh",
                                 linear.output = FALSE
        )
)

plot(cardio_nn_1)

### Predict training set

maxidx <- function(arr) {
        return(which(arr == max(arr)))
}

mypredict2_train <- compute(cardio_nn_1, cardio_train[-c(11, 12, 13, 14)])$net.result

idx_train <- apply(mypredict2_train, 1, maxidx)
prediction_train <- c("1", "2", "3")[idx_train]

table(prediction_train, cardio_train$NSP)

cardio_nn_1_train_acc <- mean(prediction_train == cardio_train$NSP)
cardio_nn_1_train_acc 

### Predict test set

mypredict2_test <- compute(cardio_nn_1, cardio_test[-c(11, 12, 13, 14)])$net.result

idx_test <- apply(mypredict2_test, 1, maxidx)
prediction_test <- c("1", "2", "3")[idx_test]

table(prediction_test, cardio_test$NSP)

cardio_nn_1_test_acc <- mean(prediction_test == cardio_test$NSP)
cardio_nn_1_test_acc 

### EM
BIC = mclustBIC(df)
summary(BIC)
plot(BIC)

fit <- Mclust(df, 3)
fit$classification

plot(fit)

summary(fit)
em_output <- as.data.frame(fit$classification)
em_output$`fit$classification` <- as.factor(em_output$`fit$classification`)

plot_ly(x= cardio$Mean, y =cardio$MSTV, z=cardio$Max, color = em_output$`fit$classification`,
        main = "EM 3 components")

#########################################################################################################

### Decision Tree
cardio5 <- cardio[-22]

# Normalize
maxs <- apply(cardio5, 2, max)
mins <- apply(cardio5, 2, min)
scaled <- as.data.frame(scale(cardio5, center = mins, scale = maxs - mins))
head(scaled)
cardio_scaled <- cbind(scaled, cardio[c(22)])

cardio_tree <- rpart(NSP ~ ., data = cardio_scaled)
plot(cardio_tree)
text(cardio_tree)

cardio6 <- cardio_scaled[c("Mean", "MSTV", "Max", "DP", "ALTV", "ASTV", "AC", "Median")]
df <- cardio6
### K-means

# Scree Plot
wss <- (nrow(df))*sum(apply(df,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters(Cardio 5 Clusters)", ylab="Within groups sum of squares") 

# K-means clustering
kc <- kmeans(df, 5)

kc$cluster <- as.factor(kc$cluster)
plot_ly(x= df$Mean, y= df$MSTV, color = kc$cluster)


### Neural Network

# Binarize the categorical output
cardio6 <- cbind(cardio6, cardio$NSP == 1)
cardio6 <- cbind(cardio6, cardio$NSP == 2)
cardio6 <- cbind(cardio6, cardio$NSP == 3)
cardio6 <- cbind(cardio6, cardio[22])

names(cardio6)[9] <- "normal"
names(cardio6)[10] <- "suspect"
names(cardio6)[11] <- "pathologic"

cols <- sapply(cardio6, is.logical)
cardio6[,cols] <- lapply(cardio6[,cols], as.numeric)

cardio7 <- cardio6

# Split data
sample2 <- sample(1:nrow(cardio7), 0.7*nrow(cardio7), replace = FALSE)
cardio_train <- cardio7[sample2, ]
cardio_test <- cardio7[-sample2, ]

# 
n <- names(cardio_train)
f <- as.formula(paste("normal + suspect + pathologic ~", 
                      paste(n[!n %in% c("normal", "suspect", "pathologic", "NSP")], 
                            collapse = " + ")))
f
set.seed(101)
system.time(
        cardio_nn_1 <- neuralnet(f,
                                 data = cardio_train, 
                                 hidden = c(5),
                                 threshold = 0.5,
                                 stepmax = 1e7,
                                 err.fct = "sse",
                                 act.fct = "tanh",
                                 linear.output = FALSE
        )
)

plot(cardio_nn_1)

### Predict training set

maxidx <- function(arr) {
        return(which(arr == max(arr)))
}

mypredict2_train <- compute(cardio_nn_1, cardio_train[-c(9, 10, 11, 12)])$net.result

idx_train <- apply(mypredict2_train, 1, maxidx)
prediction_train <- c("1", "2", "3")[idx_train]

table(prediction_train, cardio_train$NSP)

cardio_nn_1_train_acc <- mean(prediction_train == cardio_train$NSP)
cardio_nn_1_train_acc 

### Predict test set

mypredict2_test <- compute(cardio_nn_1, cardio_test[-c(9, 10, 11, 12)])$net.result

idx_test <- apply(mypredict2_test, 1, maxidx)
prediction_test <- c("1", "2", "3")[idx_test]

table(prediction_test, cardio_test$NSP)

cardio_nn_1_test_acc <- mean(prediction_test == cardio_test$NSP)
cardio_nn_1_test_acc 

### EM
BIC = mclustBIC(df)
summary(BIC)
plot(BIC)

fit <- Mclust(df, 3)
fit$classification

plot(fit)

summary(fit)
em_output <- as.data.frame(fit$classification)
em_output$`fit$classification` <- as.factor(em_output$`fit$classification`)

plot_ly(x= cardio$Mean, y =cardio$MSTV, z=cardio$Max, color = em_output$`fit$classification`,
        main = "EM 3 components")
