# Load data
training <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

cat("Training:", dim(training), "| Testing:", dim(testing))

# Remove columns with >95% missing values
na_percentage <- colMeans(is.na(training))
training_clean <- training[, na_percentage < 0.95]
testing_clean <- testing[, na_percentage < 0.95]

# Remove non-predictive columns
non_preds <- c("X", "user_name", "raw_timestamp_part_1", 
               "raw_timestamp_part_2", "cvtd_timestamp", 
               "new_window", "num_window")
training_clean <- training_clean[, !names(training_clean) %in% non_preds]
testing_clean <- testing_clean[, !names(testing_clean) %in% non_preds]

cat("After cleaning - Training:", dim(training_clean), "| Testing:", dim(testing_clean))

# Split data
set.seed(1234)
train_index <- createDataPartition(training_clean$classe, p = 0.7, list = FALSE)
train_data <- training_clean[train_index, ]
valid_data <- training_clean[-train_index, ]

cat("Train:", nrow(train_data), "| Validation:", nrow(valid_data), "| Test:", nrow(testing_clean))

# Train Random Forest
start_time <- Sys.time()
rf_model <- randomForest(classe ~ ., 
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)
training_time <- Sys.time() - start_time

cat("Training time:", round(training_time, 2), "minutes")

# Predict on validation set
pred_valid <- predict(rf_model, newdata = valid_data)
conf_matrix <- confusionMatrix(pred_valid, valid_data$classe)

cat("Accuracy:", round(conf_matrix$overall["Accuracy"], 4))
cat("\nOut-of-sample error:", round(1 - conf_matrix$overall["Accuracy"], 4))

# Get top predictors
var_imp <- importance(rf_model)
top_vars <- head(var_imp[order(-var_imp[, "MeanDecreaseGini"]), ], 10)

# Plot
barplot(rev(top_vars[, "MeanDecreaseGini"]),
        names.arg = rev(rownames(top_vars)),
        horiz = TRUE,
        las = 1,
        main = "Top 10 Predictors",
        xlab = "Importance")

# Predict on test set
test_pred <- predict(rf_model, newdata = testing_clean)

cat("Test Predictions:\n")
for(i in 1:20) {
    cat(sprintf("Problem %2d: %s\n", i, test_pred[i]))
}


## 2. **R Script Dosyası (.R)** - Prediction için

`make_predictions.R`:
```r
# Practical Machine Learning - Prediction Script
# This script generates the 20 prediction files for Coursera

# Load model
rf_model <- readRDS("random_forest_model.rds")

# Load test data
testing <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

# Preprocess test data (same as training)
na_percentage <- colMeans(is.na(training))
testing_clean <- testing[, na_percentage < 0.95]

non_preds <- c("X", "user_name", "raw_timestamp_part_1", 
               "raw_timestamp_part_2", "cvtd_timestamp", 
               "new_window", "num_window")
testing_clean <- testing_clean[, !names(testing_clean) %in% non_preds]

# Make predictions
predictions <- predict(rf_model, newdata = testing_clean)

# Create prediction files
pml_write_files <- function(x) {
    for(i in 1:length(x)) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, 
                   quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}

pml_write_files(predictions)
cat("20 prediction files created.\n")


