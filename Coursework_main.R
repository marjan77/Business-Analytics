# ************************************************
# This work is licensed under Group 7 

# ************************************************
#  PRATICAL BUSINESS ANALYTICS 2021
#  COM3018 / COMM053
#
# University of Surrey
# GUILDFORD
# Surrey GU2 7XH
#
# 15 NOVEMBER 2021
#
# UPDATE
# 1.00      15/11/2021    Initial Version v0.1 - Preprocessing
# 1.01      19/11/2021    v0.2 - Logisitic   
# 1.02      21/11/2021    v0.3 - Decision Tree
# 1.03      23/11/2021    v0.4 - Random Forest
# 1.04      24/11/2021    v0.5 - Neural network
# 1.05      26/11/2021    v1.0 - Integration
# ************************************************


#  clears all objects in "global environment"
rm(list=ls())
set.seed(42)

#Set Margin for plots
par(mar = c(4,0,4,0))
par(pty = "s")

# ************************************************
# Global Environment variables

DATASET_FILENAME  <- "AB_NYC_2019.csv"          # Name of input dataset file
OUTPUT_FIELD      <- "price_class"                    # Field name of the output class to predict

HOLDOUT           <- 70                         # % split to create TRAIN dataset


                    # Confidence p-value for outlier detection
# Set to negative means analyse but do not replace outliers



CLASS_0           <- 0                    # Resonably priced
CLASS_1           <- 1                    # Over priced



# Define and then load the libraries used in this project
# Library from CRAN     Version
# pacman	               0.5.1
# outliers	             0.14
# corrplot	             0.84
# MASS	                 7.3.51.4
# pROC	                 1.15.3
# formattable 	         0.2.01
# stats                  3.6.1
# PerformanceAnalytics   1.5.3

MYLIBRARIES<-c("outliers",
               "corrplot",
               "MASS",
               "pROC",
               "formattable",
               "stats",
               "caret",
               "PerformanceAnalytics",
               "StatMeasures",
               "C50",
               "h2o",
               "ggplot2"
               )


# Loads the libraries
library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)

# clears the console area
cat("\014")
source("user_functions.R")
# ************************************************************************************

   

raw_data <- readDataset(DATASET_FILENAME)  # data set contains 48895 records and 16 variables
model_input <- preprocess_data(raw_data)
training_records<-round(nrow(model_input)*(HOLDOUT/100))
training_data <- model_input[1:training_records,]
testing_data = model_input[-(1:training_records),]
  
# Pass the data to different models
#Logistic Model
logistic_regression(training_data,testing_data)

#Remove the output field from training data
#Decision Tree Model
training_predictors <- subset(training_data, select=-c(price_class))
training_data_class <- factor(training_data$price_class)
DT_mod <- C50::C5.0(training_predictors,training_data_class,rules = FALSE,trials = 1)
eval_decision_tree(DT_mod,testing_data,heading ="DT C5.0 ",boost = 1)

  
#Boosted Decision Tree
DT_mod <- C50::C5.0(training_predictors,training_data_class,rules = FALSE,trials = 20)
eval_decision_tree(DT_mod,testing_data,heading ="DT C5.0 Boosted ",boost = 20)

# Random Forest
randomForest_model(training_data.testing_data,numtree=500,m=sqrt(ncol(training_data)-1))
  
#Deep Neural Network
h2o.init()
NN_model(training_data,testing_data)
h2o.shutdown(prompt =FALSE)




