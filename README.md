# Practical Machine Learning Course Project

## Weight Lifting Exercise Quality Prediction

### Project Overview
This project predicts the quality of weight lifting exercises using accelerometer data from 6 participants. The goal is to classify exercises into 5 categories (A=correct, B-E=incorrect variations).

### Files in Repository

#### Main Files:
- `course_project.Rmd` - Complete analysis with code and explanations
- `course_project.html` - Compiled HTML report (for peer review)
- `make_predictions.R` - Script to generate prediction files
- `random_forest_model.rds` - Trained Random Forest model

#### Prediction Files (for Coursera Quiz):
- `problem_id_1.txt` to `problem_id_20.txt` - 20 test predictions

#### Supporting Files:
- `pml-training.csv` - Training data (if included)
- `pml-testing.csv` - Test data (if included)

### How to Reproduce

1. **Install required packages:**
   ```r
   install.packages(c("caret", "randomForest", "ggplot2", "dplyr"))
