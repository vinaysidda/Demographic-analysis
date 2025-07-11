setwd('C:/Users/ILLA RAHUL/Downloads/DAPM_Final_Model')
df = read.csv('deathRate_csv.csv')
sum(is.na(df)) #check missing values if there
library(ggplot2)
library(caret)
# Set a seed for reproducibility
set.seed(123)
# Create an index for the training set (70% of the data)
train_index <- createDataPartition(df$death_rate, p = 0.7, list = FALSE)
# Create the training set
train_data <- df[train_index, ]
# Create the test set
test_data <- df[-train_index, ]
set.seed(123)
################EDA####################
library(ggplot2)
library(dplyr)
#install.packages("ggplot2", dependencies = TRUE)
# Function to create box plot with annotations for a specific age group
create_boxplot <- function(df, age_group,cause) {
  
  # Subset data for the specified age group
  subset_data <- df[df$location_name %in% c("High SDI", "Middle SDI", "Low SDI") & df$age_name == age_group & df$cause_name == cause, ]

  
  # Create boxplot with annotations
  ggplot(subset_data, aes(x = location_name, y = death_rate, fill = sex_name)) +
    geom_boxplot() +
    labs(
      title = paste("Death Rates for SDI Levels in the", age_group, "Age Group, by Sex and cause being ",cause ),
      x = "SDI Level", y = "Death Rate", fill = "Sex"
    ) +
    theme_minimal() 
    
}

# Example usage
age_group <- "15-49 years"
age_group1 <- "0-14 years"
age_group2 <- "50-74 years"
age_group3 <- "75+ years"
create_boxplot(df, age_group,"Injuries")
create_boxplot(df, age_group1,"Injuries")
create_boxplot(df, age_group2,"Injuries")
create_boxplot(df, age_group3,"Injuries")
create_boxplot(df, age_group,"Communicable, maternal, neonatal, and nutritional diseases")
create_boxplot(df, age_group1,"Communicable, maternal, neonatal, and nutritional diseases")
create_boxplot(df, age_group2,"Communicable, maternal, neonatal, and nutritional diseases")
create_boxplot(df, age_group3,"Communicable, maternal, neonatal, and nutritional diseases")
create_boxplot(df, age_group,"Non-communicable diseases")
create_boxplot(df, age_group1,"Non-communicable diseases")
create_boxplot(df, age_group2,"Non-communicable diseases")
create_boxplot(df, age_group3,"Non-communicable diseases")
###########################################
# Load necessary libraries if not already loaded
library(ggplot2)
library(dplyr)

# Function to create a separate bar chart for each cause
create_bar_chart_for_cause <- function(df, cause) {
  # Filter data for the specified cause
  filtered_data <- df[df$cause_name == cause, ]
  
  # Create a bar chart
  ggplot(filtered_data, aes(x = age_name, y = death_rate, fill = sex_name)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
      title = paste("Death Rates by Age Group for Cause:", cause),
      x = "Age Group", y = "Death Rate", fill = "Sex"
    ) +
    theme_minimal() 
}


  create_bar_chart_for_cause(df, "Injuries")
  create_bar_chart_for_cause(df, "Communicable, maternal, neonatal, and nutritional diseases")
  create_bar_chart_for_cause(df, "Non-communicable diseases")
  

#######################linear model#########
lm_model <- lm(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = train_data)
# Print the summary of the linear model
summary(lm_model)
# Predict on the test data
# Set a seed for reproducibility
set.seed(123)
predictions_LinearModel <- predict(lm_model, newdata = test_data)
# Calculate RMSE (Root Mean Squared Error)
rmse_LinearModel <- sqrt(mean((test_data$death_rate - predictions_LinearModel)^2))
cat("RMSE:", rmse_LinearModel, "\n")
# Calculate R-squared
r_squared_LinearModel <- cor(predictions_LinearModel, test_data$death_rate)^2
cat("R-squared:", r_squared_LinearModel, "\n")
################RIDGE AND LASSO######################################
library(glmnet)

# Convert data to matrix format (required by glmnet)
X_train <- model.matrix(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = train_data)[,-1]
y_train <- train_data$death_rate

X_test <- model.matrix(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = test_data)[,-1]
y_test <- test_data$death_rate

# Fit Ridge regression model
ridge_model <- glmnet(X_train, y_train, alpha = 0)  

# Predictions on the test data for Ridge
ridge_predictions <- predict(ridge_model, s = 0.01, newx = X_test)  

# Calculate RMSE for Ridge
rmse_ridge <- sqrt(mean((y_test - ridge_predictions)^2))
cat("Ridge Regression RMSE:", rmse_ridge, "\n")

# Calculate R-squared for Ridge
r_squared_ridge <- cor(ridge_predictions, y_test)^2
cat("Ridge Regression R-squared:", r_squared_ridge, "\n")

# Fit Lasso regression model
lasso_model <- glmnet(X_train, y_train, alpha = 1)  # alpha = 1 corresponds to Lasso regression

# Predictions on the test data for Lasso
lasso_predictions <- predict(lasso_model, s = 0.01, newx = X_test)  

# Calculate RMSE for Lasso
rmse_lasso <- sqrt(mean((y_test - lasso_predictions)^2))
cat("Lasso Regression RMSE:", rmse_lasso, "\n")

# Calculate R-squared for Lasso
r_squared_lasso <- cor(lasso_predictions, y_test)^2
cat("Lasso Regression R-squared:", r_squared_lasso, "\n")
#########################Tree Model####################
library(tree)
library(caret)

# Set a seed for reproducibility
set.seed(123)

# Create a regression tree model
tree_model <- tree(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = train_data)

# Predictions on the test data
tree_predictions <- predict(tree_model, newdata = test_data)

# Calculate RMSE (Root Mean Squared Error)
rmse_tree <- sqrt(mean((test_data$death_rate - tree_predictions)^2))
cat("Decision Tree RMSE:", rmse_tree, "\n")

# Calculate R-squared
r_squared_tree <- cor(tree_predictions, test_data$death_rate)^2
cat("Decision Tree R-squared:", r_squared_tree, "\n")

########PRUNED TREE########
# Perform cross-validation for pruning
pruned_tree_model <- cv.tree(tree_model)
# Choose the best tree size
best_tree <- prune.tree(tree_model, best = pruned_tree_model$size[which.min(pruned_tree_model$dev)])
# Predictions on the test data using the pruned tree
pruned_tree_predictions <- predict(best_tree, newdata = test_data)
# Calculate RMSE and R-squared for the pruned tree
rmse_pruned_tree <- sqrt(mean((test_data$death_rate - pruned_tree_predictions)^2))
r_squared_pruned_tree <- cor(pruned_tree_predictions, test_data$death_rate)^2
cat("Pruned Decision Tree RMSE:", rmse_pruned_tree, "\n")
cat("Pruned Decision Tree R-squared:", r_squared_pruned_tree, "\n")

###############RANDOM FOREST#####################
# Fit a Random Forest regression model on the training data
library(randomForest)
# Set a seed for reproducibility
set.seed(123)
rf_model <- randomForest(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = train_data)

# Predictions on the test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Calculate RMSE (Root Mean Squared Error) for Random Forest
rmse_rf <- sqrt(mean((test_data$death_rate - rf_predictions)^2))
cat("Random Forest RMSE:", rmse_rf, "\n")

# Calculate R-squared for Random Forest
r_squared_rf <- cor(rf_predictions, test_data$death_rate)^2
cat("Random Forest R-squared:", r_squared_rf, "\n")
#################GBM MODEL###################
# Install and load the gbm package
#install.packages("gbm")
library(gbm)
set.seed(123)
# Fit a gradient boosting model
boost_model <- gbm(death_rate ~ location_id + sex_id + age_id + cause_id + year, data = train_data, distribution = "gaussian", n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)

# Predictions on the test data
boost_predictions <- predict(boost_model, newdata = test_data, n.trees = 1000)

# Calculate RMSE for Boosting
rmse_boost <- sqrt(mean((test_data$death_rate - boost_predictions)^2))
cat("Boosting RMSE:", rmse_boost, "\n")

# Calculate R-squared for Boosting
r_squared_boost <- cor(boost_predictions, test_data$death_rate)^2
cat("Boosting R-squared:", r_squared_boost, "\n")





######################
# Extract variable importance
var_importance <- summary(boost_model, plot = FALSE)$rel.inf

# Get the variable names from the boosting model summary
variable_names <- summary(boost_model, plot = FALSE)$var
# Combine variables and var_importance into a data frame
importance_data <- data.frame(Variable = variable_names, Importance = var_importance)

# Order the data frame by Importance in descending order
importance_data <- importance_data[order(importance_data$Importance, decreasing = TRUE), ]

# Plot variable importance
# Install and load necessary packages if not already installed
# install.packages("ggplot2")
library(ggplot2)

# Create a bar plot of variable importance
ggplot(importance_data, aes(x = Importance, y = Variable)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Variable Importance Plot",
       x = "Relative Importance",
       y = "Variable") +
  theme_minimal() +  # Optional: Use a minimal theme for a cleaner look
  theme(axis.text.y = element_text(hjust = 0), plot.title = element_text(hjust = 0.5)) +  # Center title
  geom_text(aes(label = round(Importance, 2)), hjust = -0.2, vjust = 0.4, color = "black", size = 3)  # Add labels to the bars

#################


# Residuals
residuals <- test_data$death_rate - boost_predictions
# Reset the layout to default
par(mfrow = c(1, 1))
# Residual Plot with Horizontal Line at y = 0
plot(boost_predictions, residuals, 
     main = "Residual Plot",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 16,  # Use solid points
     col = "blue",  # Set point color
     cex = 1.2)  # Increase point size

# Add a horizontal line at y = 0
abline(h = 0, col = "red", lwd = 2)

# Fitted vs. Actual Plot
plot(test_data$death_rate, boost_predictions, 
     main = "Fitted vs. Actual",
     xlab = "Actual Death Rates",
     ylab = "Fitted Death Rates",
     pch = 16,
     col = "blue",
     cex = 1.2)

# Add a diagonal reference line to Fitted vs. Actual Plot
abline(0, 1, col = "red", lwd = 2)

# Normal Q-Q Plot
qqnorm(residuals, 
       main = "Normal Q-Q Plot",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles",
       pch = 16,
       col = "blue",
       cex = 1.2)
qqline(residuals, col = 2, lwd = 2)





