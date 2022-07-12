

# To manually set a field type
# This will store $name=field name, $type=field type
manualTypes <- data.frame()

# ************************************************
# getClassifications() :
#
# Put in test dataset and get out class predictions of the neural network model using h2o
# Determine the threshold, plot the results and calculate metrics
#
# INPUT   :   object         - myTree       - tree
#         :   Data Frame     - testDataset  - dataset to evaluate
#         :   string         - title        - string to plot as the chart title
#         :   int            - classLabel   - lable given to the positive (TRUE) class
#         :   boolean        - plot         - TRUE to output results/charts
#
# OUTPUT  :   List       - Named evaluation measures
#
# ************************************************
getClassifications<-function(mod,
                             testDataset,
                             title,
                             classLabel=1,
                             plot=TRUE){
  
  positionClassOutput=which(names(testDataset)==OUTPUT_FIELD)
  
  #test data: dataframe with with just input fields
  test_inputs<-testDataset[,-positionClassOutput]
  
  # Generate class membership probabilities
  # Column 1 is for class 0 (bad loan) and column 2 is for class 1 (good loan)
  data_h2o <- as.h2o(testDataset)
  testPredictedClassProbs<-h2o.predict(mod,data_h2o)
  testPredictedClassProbs <- as.data.frame(testPredictedClassProbs)
  # Get the column index with the class label
  #classIndex<-which(as.numeric(colnames(testPredictedClassProbs))==classLabel)
  
  # Get the probabilities for classifying the good loans
  test_predictedProbs<-testPredictedClassProbs$p1
  
  #test data: vector with just the expected output class
  test_expected<-testDataset[,positionClassOutput]
  
  measures<-determineThreshold(test_expected=test_expected,
                               test_predicted=test_predictedProbs,
                               plot=plot,
                               title=title)
  
  #  if (plot==TRUE)
  #    NprintMeasures(results=measures,title=title)
  
  return(measures)
} #endof getClassifications()

# ************************************************
# getTreeClassifications() :
#
# Put in test dataset and get out class predictions 
# Determine the threshold, plot the results and calculate metrics
#
# INPUT   :   object         - myTree       - tree
#         :   Data Frame     - testDataset  - dataset to evaluate
#         :   string         - title        - string to plot as the chart title
#         :   int            - classLabel   - lable given to the positive (TRUE) class
#         :   boolean        - plot         - TRUE to output results/charts
#
# OUTPUT  :   List       - Named evaluation measures
#
# ************************************************
getTreeClassifications<-function(myTree,
                                 testDataset,
                                 title,
                                 classLabel=1,
                                 plot=TRUE){
  
  positionClassOutput=which(names(testDataset)==OUTPUT_FIELD)
  
  #test data: dataframe with with just input fields
  test_inputs<-testDataset[,-positionClassOutput]
  
  # Generate class membership probabilities
  # Column 1 is for class 0 (bad loan) and column 2 is for class 1 (good loan)
  
  testPredictedClassProbs<-predict(myTree,test_inputs, type="prob")
  
  # Get the column index with the class label
  classIndex<-which(as.numeric(colnames(testPredictedClassProbs))==classLabel)
  
  # Get the probabilities for classifying the good loans
  test_predictedProbs<-testPredictedClassProbs[,classIndex]
  
  #test data: vector with just the expected output class
  test_expected<-testDataset[,positionClassOutput]
  
  measures<-determineThreshold(test_expected=test_expected,
                                test_predicted=test_predictedProbs,
                                plot=plot,
                                title=title)
  
  #  if (plot==TRUE)
  #    NprintMeasures(results=measures,title=title)
  
  return(measures)
} #endof getTreeClassifications()

# ************************************************
# auroc() :
#
# Calculate the Area Under Curve (AUC) for ROC
#
# INPUT   :   vector double     - score            - probability of being class 1
#             vector double     - bool             - Expected class of 0 or 1
#
# OUTPUT  :   double   - AUC
#
# ************************************************
# By Miron Kursa https://mbq.me
# See https://stackoverflow.com/questions/4903092/calculate-auc-in-r

auroc <- function(score, bool) {
  n1 <- sum(!bool)
  n2 <- sum(bool)
  U  <- sum(rank(score)[!bool]) - n1 * (n1 + 1) / 2
  return(1 - U / n1 / n2)
}

# ************************************************
# calcMeasures() :
#
# Evaluation measures for a confusion matrix
#
# INPUT: numeric  - TP, FN, FP, TN
#
# OUTPUT: A list with the following entries:
#        TP        - double - True Positive records
#        FP        - double - False Positive records
#        TN        - double - True Negative records
#        FN        - double - False Negative records
#        accuracy  - double - accuracy measure
#        pgood     - double - precision for "good" (values are 1) measure
#        pbad      - double - precision for "bad" (values are 1) measure
#        FPR       - double - FPR measure
#        TPR       - double - FPR measure
#        TNR       - double - TNR measure
#        MCC       - double - Matthew's Correlation Coeficient
#
# 080819NRT added TNR measure
# ************************************************
calcMeasures<-function(TP,FN,FP,TN){
  
  retList<-list(  "TP"=TP,
                  "FN"=FN,
                  "TN"=TN,
                  "FP"=FP,
                  "accuracy"=100.0*((TP+TN)/(TP+FP+FN+TN)),
                  "pgood"=   100.0*(TP/(TP+FP)),
                  "pbad"=    100.0*(TN/(FN+TN)),
                  "FPR"=     100.0*(FP/(FP+TN)),
                  "TPR"=     100.0*(TP/(TP+FN)),
                  "TNR"=     100.0*(TN/(FP+TN)),
                  "MCC"=     ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  )
  return(retList)
}

# ************************************************
# calcConfusion() :
#
# Calculate a confusion matrix for 2-class classifier
# INPUT: vector - expectedClass  - {0,1}, Expected outcome from each row (labels)
#        vector - predictedClass - {0,1}, Predicted outcome from each row (labels)
#
# OUTPUT: A list with the  entries from NcalcMeasures()
#
# 070819NRT convert values to doubles to avoid integers overflowing
# Updated to the following definition of the confusion matrix
#
#                    ACTUAL
#               ------------------
# PREDICTED     FRAUD   |  GENUINE
#               ------------------
#     FRAUD      TP     |    FP
#               ==================
#     GENUINE    FN     |    TN
#
#
# ************************************************
calcConfusion<-function(expectedClass,predictedClass){
  
  confusion<-table(factor(predictedClass,levels=0:1),factor(expectedClass,levels=0:1))
  
  # This "converts" the above into our preferred format
  
  TP<-as.double(confusion[2,2])
  FN<-as.double(confusion[1,2])
  FP<-as.double(confusion[2,1])
  TN<-as.double(confusion[1,1])
  
  return(calcMeasures(TP,FN,FP,TN))
  
} #endof calcConfusion()

# ************************************************
# EvaluateClassifier() :
#
# Use dataset to generate predictions from model
# Evaluate as classifier using threshold value
#
# INPUT   :   vector double     - probs        - probability of being class 1
#             Data Frame        - testing_data - Dataset to evaluate
#             double            - threshold     -cutoff (probability) for classification
#
# OUTPUT  :   List       - Named evaluation measures
#                        - Predicted class probability
#
# ************************************************
EvaluateClassifier<-function(test_predicted,test_expected,threshold) {
  
  predictedClass<-ifelse(test_predicted<threshold,0,1)
  
  results<-calcConfusion(expectedClass=test_expected,
                          predictedClass=predictedClass)
  
  return(results)
} #endof EvaluateClassifier()


# ************************************************
# determineThreshold() :
#
# For the range of threholds [0,1] calculate a confusion matrix
# and classifier metrics.
# Deterime "best" threshold based on either distance or Youdan
# Plot threshold chart and ROC chart
#
# Plot the results
#
# INPUT   :   vector double  - test_predicted   - probability of being class 1
#         :   vector double  - test_expected    - dataset to evaluate
#         :   boolean        - plot             - TRUE=output charts
#         :   string         - title            - chart title
#
# OUTPUT  :   List       - Named evaluation measures from confusion matrix
#                        - Threshold at min Euclidean distance
#                        - AUC - area under the ROC curve
#                        - Predicted class probability
#
# 241019NRT - added plot flag and title for charts
# 311019NRT - added axis bound checks in abline plots
# 191020NRT - Updated to use own ROC plot & calculate AUC
# ************************************************
determineThreshold<-function(test_predicted,
                              test_expected,
                              plot=TRUE,
                              title=""){
  toPlot<-data.frame()
  
  #Vary the threshold
  for(threshold in seq(0,1,by=0.01)){
    results<-EvaluateClassifier(test_predicted=test_predicted,
                                 test_expected=test_expected,
                                 threshold=threshold)
    toPlot<-rbind(toPlot,data.frame(x=threshold,fpr=results$FPR,tpr=results$TPR))
  }
  
  # the Youden index is the vertical distance between the 45 degree line
  # and the point on the ROC curve.
  # Higher values of the Youden index are better than lower values.
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5082211/
  # Youdan = sensitivty + specificity -1
  #        = TPR + (1-FPR) -1
  
  toPlot$youdan<-toPlot$tpr+(1-toPlot$fpr)-1
  
  # 121020NRT - max Youdan
  # use which.max() to return a single index to the higest value in the vector
  maxYoudan<-toPlot$x[which.max(toPlot$youdan)]
  
  # Euclidean distance sqrt((1 − sensitivity)^2+ (1 − specificity)^2)
  # To the top left (i.e. perfect classifier)
  toPlot$distance<-sqrt(((100-toPlot$tpr)^2)+((toPlot$fpr)^2))
  
  # 121020NRT - Euclidean distance to "perfect" classifier (smallest the best)
  # use which.min() to return a single index to the lowest value in the vector
  minEuclidean<-toPlot$x[which.min(toPlot$distance)]
  
  # ************************************************
  # Plot threshold graph
  
  if (plot==TRUE){
    # Sensitivity (TPR)
    plot(toPlot$x,toPlot$tpr,
         xlim=c(0, 1), ylim=c(0, 100),
         type="l",lwd=3, col="blue",
         xlab="Threshold",
         ylab="%Rate",
         main=paste("Threshold Perfomance Classifier Model",title))
    
    # Plot the specificity (1-FPR)
    lines(toPlot$x,100-toPlot$fpr,type="l",col="red",lwd=3,lty=1)
    
    # The point where specificity and sensitivity are the same
    crosspoint<-toPlot$x[which(toPlot$tpr<(100-toPlot$fpr))[1]]
    
    if (!is.na(crosspoint)){
      if ((crosspoint<1) & (crosspoint>0))
        abline(v=crosspoint,col="red",lty=3,lwd=2)
    }
    
    # Plot the Euclidean distance to "perfect" classifier (smallest the best)
    lines(toPlot$x,toPlot$distance,type="l",col="green",lwd=2,lty=3)
    
    # Plot the min distance, as might be more (311019NRT check it is within range)
    if ((minEuclidean<1) & (minEuclidean>0))
      abline(v=minEuclidean,col="green",lty=3,lwd=2)
    
    # Youdan (Vertical distance between the 45 degree line and the point on the ROC curve )
    lines(toPlot$x,toPlot$youdan,type="l",col="purple",lwd=2,lty=3)
    
    if ((maxYoudan<1) & (maxYoudan>0))
      abline(v=maxYoudan,col="purple",lty=3,lwd=2)
    
    legend("bottom",c("TPR","1-FPR","Distance","Youdan"),col=c("blue","red","green","purple"),lty=1:2,lwd=2)
    text(x=0,y=50, adj = c(-0.2,2),cex=1,col="black",paste("THRESHOLDS:\nEuclidean=",minEuclidean,"\nYoudan=",maxYoudan))
    
    # ************************************************
    # 121020NRT ROC graph
    
    sensitivityROC<-toPlot$tpr[which.min(toPlot$distance)]
    specificityROC<-100-toPlot$fpr[which.min(toPlot$distance)]
    auc<-auroc(score=test_predicted,bool=test_expected) # Estimate the AUC
    
    # Set origin point for plotting
    toPlot<-rbind(toPlot,data.frame(x=0,fpr=0,tpr=0, youdan=0,distance=0))
    
    plot(100-toPlot$fpr,toPlot$tpr,type="l",lwd=3, col="black",
         main=paste("ROC:",title),
         xlab="Specificity (1-FPR) %",
         ylab="Sensitivity (TPR) %",
         xlim=c(100,0),
         ylim=c(0,100)
    )
    
    axis(1, seq(0.0,100,10))
    axis(2, seq(0.0,100,10))
    
    #Add crosshairs to the graph
    abline(h=sensitivityROC,col="red",lty=3,lwd=2)
    abline(v=specificityROC,col="red",lty=3,lwd=2)
    
    annotate<-paste("Threshold: ",round(minEuclidean,digits=4L),
                    "\nTPR: ",round(sensitivityROC,digits=2L),
                    "%\n1-FPR: ",round(specificityROC,digits=2L),
                    "%\nAUC: ",round(auc,digits=2L),sep="")
    
    
    text(x=specificityROC, y=sensitivityROC, adj = c(-0.2,1.2),cex=1, col="red",annotate)
    
  } # endof if plotting
  
  # Select the threshold - I have choosen distance
  
  myThreshold<-minEuclidean      # Min Distance should be the same as analysis["threshold"]
  
  #Use the "best" distance threshold to evaluate classifier
  results<-EvaluateClassifier(test_predicted=test_predicted,
                               test_expected=test_expected,
                               threshold=myThreshold)
  
  results$threshold<-myThreshold
  results$AUC<-auroc(score=test_predicted,bool=test_expected) # Estimate the AUC
  
  return(results)
} 

# calculate_MCC() :
#
# Calculate Matthews correlation coefficient of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

calculate_MCC<- function(TP,TN,FP,FN)
{
  mcc <- ((TP * TN) - (FP *FN) ) / (sqrt((TP + FN)*(TP + FP)) *sqrt((TN + FP) * (TN + FN)))
  # a <- (sqrt((TP + FN)*(TP + FP)))
  print(paste("MCC of the model :",mcc))
  #return(mcc)
  
}

# ************************************************
# get_accuracy() :
#
# Calculate accuracy of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

get_accuracy <- function(TP,TN,FP,FN)
{
  accuracy <- (TP+TN)/(TP + FP + TN +FN)
  print(paste("Accuracy of the model : ",round(accuracy,digits = 4)*100,"%"))
  
}

# ************************************************

# classification_error_rate() :
#
# Calculate Classification_error_rate of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

classification_error_rate <- function(TP,TN,FP,FN)
{
  error_rate  <- (FP+FN)/(TP + FP + TN +FN)
  print(paste("Classification error rate of the model : ",round(error_rate,digits = 4)*100,"%"))
  
}

# ************************************************
# get_precision() :
#
# Calculate precision of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

get_precision <- function(TP,TN,FP,FN)
{
  precision  <- (TP)/(TP + FP )
  print(paste("Precision of the model : ",round(precision,digits = 4)))
  return(precision)
}

# ************************************************

# get_sensitivity() :
#
# Calculate sensitivity of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

get_sensitivity <- function(TP,TN,FP,FN)
{
  sensitivity  <- (TP)/(TP + FN )
  print(paste("Sensitivity of the model : ",round(sensitivity,digits = 4)))
  return(sensitivity)
}

# ************************************************
# get_specificity() :
#
# Calculate specificity of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

get_specificity<- function(TP,TN,FP,FN)
{
  specificity  <- (TN)/(TN + FP )
  print(paste("specificity of the model : ",round(specificity,digits = 4)))
  
}

# ************************************************

# calculate_f_score() :
#
# Calculate f-score of model 
#
# INPUT:
#        Numeric
#
# OUTPUT : Null
# ************************************************

Calculate_f_score<- function(precision,sensitivity)
{
  f_score  <- (2*precision*sensitivity)/(precision+sensitivity )
  print(paste("F_Score of the model : ",round(f_score,digits = 4)))
  
}



# ************************************************
# PREPROCESSING_setInitialFieldType() :
#
# Set  each field for NUMERIC or SYNBOLIC
#
# INPUT:
#        String - name - name of the field to manually set
#        String - type - manual type
#
# OUTPUT : None
# ************************************************
PREPROCESSING_setInitialFieldType<-function(name,type){
  
  #Sets in the global environment
  manualTypes<<-rbind(manualTypes,data.frame(name=name,type=type,stringsAsFactors = FALSE))
}


# ************************************************
# PREPROCESSING_encodeOrderedSymbols()
#
# Encodes symbols in a field to 1/n
# See Lecture 3, slide 48, Ordered Catagorical encoding
#
# INPUT:    Data Frame     - originalDataset - dataset to preprocess
#           String         - fieldName       - name of the field
#           String Vector  - orderedSymbols  - list of symbols in order to be encoded
#
# OUTPUT :  Data Frame - updated dataset
# 111119NRT - minor bug with length() compare
# ************************************************
PREPROCESSING_encodeOrderedSymbols<-function(dataset, fieldName, orderedSymbols){
  TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
  TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
  TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
  TYPE_IGNORE       <- "IGNORE"             # field is not encoded
  
  PREPROCESSING_setInitialFieldType(fieldName, TYPE_IGNORE)
  
  field<-which(names(dataset)==fieldName)
  
  for (eachSymbol in 1:length(orderedSymbols)){
    records<-which(dataset[,field]==orderedSymbols[eachSymbol])
    if (length(records)>0){
      dataset[records,field]<-(eachSymbol-1)/(length(orderedSymbols)-1)
    }
  }
  dataset[,field]<-as.numeric(dataset[,field])
  return(dataset)
}


# ************************************************
# gencorplot() :
#
# Generate correlation plot
#
# INPUT:
#        Data Frame 
#
# OUTPUT : Data frame
# ************************************************

gencorplot <- function(data)
{
  
  symbols <- unique(data$neighbourhood)
  plt_data <- PREPROCESSING_encodeOrderedSymbols(data,"neighbourhood",symbols)
  symbols <- unique(plt_data$neighbourhoodgroup)
  plt_data <- PREPROCESSING_encodeOrderedSymbols(plt_data,"neighbourhoodgroup",symbols)
  symbols <- unique(plt_data$roomtype)
  plt_data <- PREPROCESSING_encodeOrderedSymbols(plt_data,"roomtype",symbols)
  
  # To generate correlation plot
  M = cor(plt_data,method = c("pearson"))
  corrplot(M,method = "color",order='alphabet')
  
  
  
}

# ************************************************
# PREPROCESSING_categorical() :
#
# Transform SYMBOLIC or DISCRETE fields using 1-hot-encoding
#
# INPUT: data frame    - dataset      - symbolic fields
#        vector string - field_types  - types per field {ORDINAL, SYMBOLIC, DISCRETE}
#
# OUTPUT : data frame    - transformed dataset
#
# 18/2/2021 NRT Updated for efficiency
# ************************************************

PREPROCESSING_categorical<-function(dataset,field_types){
  TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
  TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
  TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
  TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
  TYPE_IGNORE       <- "IGNORE"             # field is not encoded
  MAX_LITERALS      <- 5                    # Maximum numner of hotcoding new fields
  
  catagorical<-data.frame()
  
  #categorical_fields<-names(dataset)[which(field_types==TYPE_SYMBOLIC | field_types==TYPE_DISCRETE)]
  categorical_fields<-names(dataset)[which(field_types==TYPE_SYMBOLIC)]
  
  # for each field
  for (field in categorical_fields){
    
    # Convert into factors. A level for each unique string
    ffield<-factor(dataset[,field])
    
    # Check if too many unique values to encode
    if (nlevels(ffield) > MAX_LITERALS) {
      #print(paste("Too many  liiterals in:",
       #           field,
        #          nlevels(ffield)))
      next
    }
    
    # Check if just one value!
    if (nlevels(ffield) ==1) {
      print(paste("Prof. Nick says - field stuck at a single value:",
                  field))
      next
    }
    
    # 1-hot encoding. A new column for each unique "level"
    xx<-data.frame(model.matrix(~ffield+0, data=ffield))
    
    names(xx)<-gsub("ffield",field,names(xx))
    
    # If 2 unique values, then can encode as a single "binary" column
    if (ncol(xx)==2){
      xx<-xx[,-2,drop=FALSE]
      names(xx)<-field  # Field name without the value appended
    }
    
    catagorical<-as.data.frame(append(catagorical,xx))
    
  } #endof for()
  return (catagorical)
  
} # endof categorical_encoding()



# ************************************************
# PREPROCESSING_discreteNumeric() :
#
# Test NUMERIC field if DISCRETE or ORDINAL
#
# INPUT: data frame      - dataset     - input data
#        vector strings  - field_types - Types per field, either {NUMERIC, SYMBOLIC}
#        int             - cutoff      - Number of empty bins needed to determine discrete (1-10)
#
# OUTPUT : vector strings - Updated with types per field {DISCRETE, ORDINAL}
# ************************************************
# Plots histogram for visulisation
# ************************************************
PREPROCESSING_discreteNumeric<-function(dataset,field_types,cutoff){
  TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
  TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
  TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
  TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
  TYPE_IGNORE       <- "IGNORE"             # field is not encoded
  
  #For every field in our dataset
  for(field in 1:(ncol(dataset))){
    
    #Only for fields that are all numeric
    if (field_types[field]==TYPE_NUMERIC) {
      
      #191020NRT use R hist() function to create 10 bins
      histogramAnalysis<-hist(dataset[,field], breaks = 10, plot=FALSE)
      bins<-histogramAnalysis$counts/length(dataset[,field])*10  # Convert to %
      
      graphTitle<-"AUTO:"
      
      #If the number of bins with less than 1% of the values is greater than the cutoff
      #then the field is deterimed to be a discrete value
      
      if (length(which(bins<1.0))>cutoff)
        field_types[field]<-TYPE_DISCRETE
      else
        field_types[field]<-TYPE_ORDINAL
      
      #Type of field is the chart name
      hist(dataset[,field], breaks = 10, plot=TRUE,
           main=paste(graphTitle,field_types[field]),
           xlab=names(dataset[field]),ylab="Number of Records",
           yaxs="i",xaxs="i",border = NA)
      
    } #endif numeric types
  } #endof for
  return(field_types)
}


# ************************************************
# PREPROCESSING_initialFieldType() :
#
# Test each field for NUMERIC or SYNBOLIC
#
# INPUT: Data Frame - dataset - data
#
# OUTPUT : Vector - Vector of types {NUMERIC, SYMBOLIC}
# ************************************************
PREPROCESSING_initialFieldType<-function(dataset){
  TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
  TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
  TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
  TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
  TYPE_IGNORE       <- "IGNORE"             # field is not encoded
  
  field_types<-vector()
  for(field in 1:(ncol(dataset))){
    
    entry<-which(manualTypes$name==names(dataset)[field])
    if (length(entry)>0){
      field_types[field]<-manualTypes$type[entry]
      next
    }
    
    if (is.numeric(dataset[,field])) {
      field_types[field]<-TYPE_NUMERIC
    }
    else {
      field_types[field]<-TYPE_SYMBOLIC
    }
  }
  return(field_types)
}


# ************************************************
# PREPROCESSING_prettyDataset()
# Output simple dataset field analysis results as a table in "Viewer"
#
# REQUIRES: formattable
#
# INPUT: data frame    - dataset, full dataset used for train/test
#                      - Each row is one record, each column in named
#                      - Values are not scaled or encoded
#        String - OPTIONAL string which is used in table as a header
#
# OUTPUT : none
#
# Requires the library: PerformanceAnalytics
#                       formattable
# ************************************************
PREPROCESSING_prettyDataset<-function(dataset,...){
  
  params <- list(...)
  
  
  tidyTable<-data.frame(Field=names(dataset),
                        Catagorical=FALSE,
                        Symbols=0,
                        Name=0,
                        Min=0.0,
                        Mean=0.0,
                        Max=0.0,
                        Skew=0.0,
                        stringsAsFactors = FALSE)
  if (length(params)>0){
    names(tidyTable)[1]<-params[1]
  }
  
  for (i in 1:ncol(dataset)){
    isFieldAfactor<-!is.numeric(dataset[,i])
    tidyTable$Catagorical[i]<-isFieldAfactor
    if (isFieldAfactor){
      tidyTable$Symbols[i]<-length(unique(dataset[,i]))  #Number of symbols in catagorical
      #Gets the count of each unique symbol
      symbolTable<-sapply(unique(dataset[,i]),function(x) length(which(dataset[,i]==x)))
      majoritySymbolPC<-round((sort(symbolTable,decreasing = TRUE)[1]/nrow(dataset))*100,digits=0)
      tidyTable$Name[i]<-paste(names(majoritySymbolPC),"(",majoritySymbolPC,"%)",sep="")
    } else
    {
      tidyTable$Max[i]<-round(max(dataset[,i]),2)
      tidyTable$Mean[i]<-round(mean(dataset[,i]),2)
      tidyTable$Min[i]<-round(min(dataset[,i]),2)
      tidyTable$Skew[i]<-round(PerformanceAnalytics::skewness(dataset[,i],method="moment"),2)
    }
  }
  
  #Sort table so that all numerics are first
  t<-formattable::formattable(tidyTable[order(tidyTable$Catagorical),],
                              list(Catagorical = formatter("span",style = x ~ style(color = ifelse(x,"green", "red")),
                                                           x ~ icontext(ifelse(x, "ok", "remove"), ifelse(x, "Yes", "No"))),
                                   Symbols = formatter("span",style = x ~ style(color = "black"),x ~ ifelse(x==0,"-",sprintf("%d", x))),
                                   Min = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Catagorical,"-",format(Min, nsmall=2, big.mark=","))),
                                   Mean = formatter("span",style = x ~ style(color = "black"),~ ifelse(Catagorical,"-",format(Mean, nsmall=2, big.mark=","))),
                                   Max = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Catagorical,"-",format(Max, nsmall=2, big.mark=","))),
                                   Skew = formatter("span",style = x ~ style(color = "black"),~ ifelse(Catagorical,"-",sprintf("%.2f", Skew)))
                              ))
  print(t)
}



# ************************************************
# get_class_details() :
#
# Set  eclass labels for the price field
#
# INPUT:
#        Data Frame 
#
# OUTPUT : Data frame
# ************************************************

get_class_details <- function(data)
{

  
  # To get class details based on room type and neighbourhood group
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Private room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Entire home/apt")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Shared room")) & (result$price <= (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_0
  # 
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Private room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Entire home/apt")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Shared room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_0
  # 
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Private room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Entire home/apt")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Shared room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_0
  # 
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Private room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Entire home/apt")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Shared room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_0
  # 
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Private room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Entire home/apt")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_0
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Shared room")) & (result$price < (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_0
  # 
  # # checking for class 1
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Private room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Entire home/apt")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Bronx") & (result$roomtype == "Shared room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Bronx") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_1
  # 
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Private room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Entire home/apt")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Brooklyn") & (result$roomtype == "Shared room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Brooklyn") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_1
  # 
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Private room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Entire home/apt")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Manhattan") & (result$roomtype == "Shared room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Manhattan") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_1
  # 
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Private room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Entire home/apt")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Queens") & (result$roomtype == "Shared room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Queens") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_1
  # 
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Private room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Private room")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Entire home/apt")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Entire home/apt")), "price"])))),"price"] <- CLASS_1
  # result[which(((result$neighbourhoodgroup == "Staten Island") & (result$roomtype == "Shared room")) & (result$price >= (mean(data[which((data$neighbourhoodgroup == "Staten Island") & (data$roomtype == "Shared room")), "price"])))),"price"] <- CLASS_1
  # names(result)[ncol(result)] <- "price_class"
  
  #To get class details based on neighbourhood group alone
  
  data_list <- split(data,f=data$neighbourhoodgroup)
  
  data_list$Bronx[which(data_list$Bronx$price <= mean(data_list$Bronx$price)),"price_class"] <- CLASS_0
  data_list$Bronx[which(data_list$Bronx$price > mean(data_list$Bronx$price)),"price_class"] <- CLASS_1
  
  data_list$Brooklyn[which(data_list$Brooklyn$price <= mean(data_list$Brooklyn$price)),"price_class"] <- CLASS_0
  data_list$Brooklyn[which(data_list$Brooklyn$price > mean(data_list$Brooklyn$price)),"price_class"] <- CLASS_1
  
  data_list$Manhattan[which(data_list$Manhattan$price <= mean(data_list$Manhattan$price)),"price_class"] <- CLASS_0
  data_list$Manhattan[which(data_list$Manhattan$price > mean(data_list$Manhattan$price)),"price_class"] <- CLASS_1
  
  data_list$Queens[which(data_list$Queens$price <= mean(data_list$Queens$price)),"price_class"] <- CLASS_0
  data_list$Queens[which(data_list$Queens$price > mean(data_list$Queens$price)),"price_class"] <- CLASS_1
  
  data_list$`Staten Island`[which(data_list$`Staten Island`$price <= mean(data_list$`Staten Island`$price)),"price_class"] <- CLASS_0
  data_list$`Staten Island`[which(data_list$`Staten Island`$price > mean(data_list$`Staten Island`$price)),"price_class"] <- CLASS_1
  
  out_data <- rbind(data_list$Bronx, data_list$Brooklyn, data_list$Manhattan, data_list$Queens, data_list$`Staten Island`)
  out_data <- subset(out_data, select = -c(price))
  
  
  # To get class  details based on room type alone   
  
  #data_list <- split(outlierFree_data,f=outlierFree_data$roomtype)
  
  #data_list$`Private room`[which(data_list$`Private room`$price <= mean(data_list$`Private room`$price)),"price_class"] <- CLASS_0
  #data_list$`Private room`[which(data_list$`Private room`$price > mean(data_list$`Private room`$price)),"price_class"] <- CLASS_1
  
  #data_list$`Entire home/apt`[which(data_list$`Entire home/apt`$price <= mean(data_list$`Entire home/apt`$price)),"price_class"] <- CLASS_0
  #data_list$`Entire home/apt`[which(data_list$`Entire home/apt`$price > mean(data_list$`Entire home/apt`$price)),"price_class"] <- CLASS_1
  
  #data_list$`Shared room`[which(data_list$`Shared room`$price <= mean(data_list$`Shared room`$price)),"price_class"] <- CLASS_0
  #data_list$`Shared room`[which(data_list$`Shared room`$price > mean(data_list$`Shared room`$price)),"price_class"] <- CLASS_1
  
  #out_data <- rbind(data_list$`Shared room`, data_list$`Private room`, data_list$`Entire home/apt`)
  #out_data <- subset(out_data, select = -c(price))
  
  return(out_data)  
}

# ************************************************
# readDataset() :
#
# Read a CSV file from working directory
#
# INPUT: string - csvFilename - CSV filename
#
# OUTPUT : data frame - contents of the headed CSV file
# ************************************************
readDataset<-function(csvFilename){
  
  dataset<-read.csv(csvFilename,encoding="UTF-8",stringsAsFactors = FALSE)
  
  # The field names "confuse" some of the library algorithms
  # As they do not like spaces, punctuation, etc.
  names(dataset)<-PREPROCESSING_removePunctuation(names(dataset))
  
  print(paste("CSV dataset",csvFilename,"has been read. Records=",nrow(dataset)))
  return(dataset)
}


# ************************************************
# PREPROCESSING_removePunctuation()
#
# INPUT: String - fieldName - name of field
#
# OUTPUT : String - name of field with punctuation removed
# ************************************************
PREPROCESSING_removePunctuation<-function(fieldName){
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}



# ************************************************
# preprocess_data()
#
# INPUT: Data Frame
#
# OUTPUT : Pre processed data frame
# ************************************************
preprocess_data <-function(raw_data)
{
  TYPE_DISCRETE     <- "DISCRETE"           # field is discrete (numeric)
  TYPE_ORDINAL      <- "ORDINAL"            # field is continuous numeric
  TYPE_SYMBOLIC     <- "SYMBOLIC"           # field is a string
  TYPE_NUMERIC      <- "NUMERIC"            # field is initially a numeric
  TYPE_IGNORE       <- "IGNORE"             # field is not encoded
  SCALE_DATASET     <- FALSE                      # Set to true to scale dataset before ML stage
  OUTLIER_CONF      <- 0.95   
  
  DISCRETE_BINS     <- 9                    # Number of empty bins to determine discrete

  

  
  #Remove the records for which price is zero - 11 records have price = 0
  raw_data <- raw_data[!(raw_data$price ==0),]
  
  # To move price field to the last coloumn
  raw_data <- subset(raw_data,select  =c(11:16,1:10))
  
  
  
  
  # Remove the fields which are irrelevant to the objective/modelling
  # Removing id, name, hostname, lastreview
  #cut_data <- subset (raw_data, select = -c(id,name,hostname,hostid,neighbourhood,numberofreviews,reviewspermonth,neighbourhoodgroup,lastreview))
  cut_data <- subset (raw_data, select = -c(id,name,hostname,hostid,lastreview))
  
  #Replace NA values in reviews per month with 0
  cut_data$reviewspermonth[is.na(cut_data$reviewspermonth)] <- 0
  
  #To remove NULL values if any - removed
  cut_data <- na.omit(cut_data)
  
  # To get box plot of price
  boxplot(cut_data$price,col = "red",border="black",ylab = "Price",main ="Boxplot for price")
  # To get box plot of price in each borough
  boxplot(cut_data$price ~ cut_data$neighbourhoodgroup,col = "red",border="black",ylab = "Price",xlab = "Newyork Boroughs",main ="Different boxplots for price per borough")
  #To get box plot of price in each room type
  boxplot(cut_data$price ~ cut_data$roomtype,col = "red",border="black",ylab = "Price",xlab = "Room types",main ="Different boxplots for price per room type")
  
  
  # Removing outlier values in price 
  outlierFree_data <-cut_data
  cutoff <- quantile(cut_data$price)
  interquartile_range = cutoff[4] - cutoff[2]
  outlierFree_data <- subset(outlierFree_data, outlierFree_data$price <= (cutoff[4] + (interquartile_range*1.5))) # 3 rd + (quartile * 1.5
  outlierFree_data <- subset(outlierFree_data, outlierFree_data$price >= (cutoff[2] - (interquartile_range*1.5))) # 1 st  - (quartile range * 1.5) 
  
  #After observing the plots and data, the below obsevations are made:
  #Since 99% of the records have reviews per month less than 10, all the records having value above 10 should be removed
  #Since 99% of the records have minimum nights less than 30, all the other records are removed
  #Since 99% of the host listing count is less than 50, all the other records are removed
  #outlierFree_data <- subset(cut_data,cut_data$reviewspermonth <= 25)
  #outlierFree_data <- subset(outlierFree_data, outlierFree_data$minimumnights <30)
  #outlierFree_data <- subset(outlierFree_data, outlierFree_data$calculatedhostlistingscount <50)
  
  #outlierFree_data <- subset(outlierFree_data,outlierFree_data$calculatedhostlistingscount <=20)
  #outlierFree_data <- subset(outlierFree_data,outlierFree_data$numberofreviews <150)
  
  # To round of latitude and longitude in the data set to two decimal places
  outlierFree_data$latitude <- round(outlierFree_data$latitude,digits = 2)
  outlierFree_data$longitude <- round(outlierFree_data$longitude,digits = 2)
  
  
  # ***************************************************************************************************
  
  
  
  # To visualize the distribution of records in the data
  tbl <- with(outlierFree_data, table(roomtype,neighbourhoodgroup))
  barplot(tbl,beside = TRUE, legend = TRUE,col=c("#a1e9f0","#b9e38d","#eb8060"),ylab = "Number of records", xlab = "Boroughs",main = "Distribution of records ")
  
  
  boxplot(outlierFree_data$price ~ outlierFree_data$roomtype,border="black",ylab = "Price",xlab = "Room types",main ="Different boxplots for price per room type")


  boxplot(outlierFree_data$price ~ outlierFree_data$neighbourhoodgroup,border="black",ylab = "Price",xlab = "Boroughs",main ="Different boxplots for price per borough")

  class_data <- get_class_details(data  = outlierFree_data )
  
  #Randomly reorder the data
  class_data <-  class_data[sample(1:nrow(class_data)),]
  
  
  # To see in a tabular format
  PREPROCESSING_prettyDataset(class_data)
  field_types<-PREPROCESSING_initialFieldType(class_data)
  
  # ************************************************
  # View the field types on the console
  
  numeric_fields<-names(class_data)[field_types=="NUMERIC"]
  print(paste("NUMERIC FIELDS=",length(numeric_fields)))
  print(numeric_fields)
  
  symbolic_fields<-names(class_data)[field_types=="SYMBOLIC"]
  print(paste("SYMBOLIC FIELDS=",length(symbolic_fields)))
  print(symbolic_fields)
  # ************************************************
  # Determine if the numeric fields might be discreet numeric
  
  field_types1<-PREPROCESSING_discreteNumeric(dataset=class_data,
                                              field_types=field_types,
                                              cutoff=DISCRETE_BINS)
  
  #Since host ID is considered as an ORDINAL
  results<-data.frame(field=names(class_data),initial=field_types,types1=field_types1)
  print(formattable::formattable(results))
  
  
  
  # ************************************************
  field_types1[length(field_types1)] <- "DISCRETE"
  # To make a numerical data set by encoding the symbolic/discrete fields
  catagoricalReadyforML<-PREPROCESSING_categorical(dataset=class_data,field_types=field_types1)
  categorical_fields <- names(class_data)[which(field_types==TYPE_SYMBOLIC)]
  categorical_fields<- categorical_fields[categorical_fields != OUTPUT_FIELD]
  print(formattable::formattable(data.frame(fields=names(catagoricalReadyforML))))
  
  # Combine the two sets of data that are ready for ML
  final_data <-cbind(catagoricalReadyforML,class_data)
  
  #Remove the fields which are encoded using 1hot encoding
  encoded_data <- final_data[,!(names(final_data) %in% categorical_fields)]
  
  # ************************************************
  
  
  #Geneate correation map after encoding the catagorical fields as ordered symbols
  
  gencorplot(outlierFree_data)
  
  #converting the price_class  field variables  into factor variable
  #class_data$price_class <- as.factor(class_data$price_class)
  
  # To visualize the distribution of price class in the data
  tbl <- with(encoded_data, table(price_class)/nrow(encoded_data)*100)
  #tbl <- with(class_data, table(price_class))
  bp <- barplot(tbl,beside = TRUE, legend = TRUE,col=c("#a1e9f0","#b9e38d","#eb8060"),ylab = "Number of records (%)", xlab = "Price class",main = "Distribution of price class before sampling ",names.arg = c("Reasonable", "Overpriced"))
  text(bp, 0,round(tbl,1),cex=1,pos=3)
  
  
  
  #Sampling to make class balanced - performing both over sampling and undersampling - each of 10%
  tbl <- table(encoded_data$price_class)
  n1 = round(abs(tbl[1]-tbl[2])*20/100)
  n1 <-  tbl[1] +tbl[2] -n1
  n2 <- n1 + round(abs(tbl[1]-tbl[2])*20/100)
  balanced_data <- encoded_data

  #n2 <- n1 + nrow(encoded_data)
  #balanced_data <- ovun.sample(price_class ~ ., data = encoded_data, method = "under", N = n1 )$data
  #balanced_data <- ovun.sample(price_class ~ ., data = balanced_data,method = "over", N = n2 )$data
  #balanced_data <- ovun.sample(price_class ~ .,data = model_input,method = "over",N=49840)$data
  #balanced_data <- ROSE(price_class ~ .,data = model_input,seed = 1)$data
  
  
  # To visualize the distribution of price class in the data after sampling
  tbl <- with(balanced_data, table(price_class)/nrow(balanced_data)*100)
  #tbl <- with(balanced_data, table(price_class))
  bp <- barplot(tbl,beside = TRUE, legend = TRUE,col=c("#a1e9f0","#b9e38d","#eb8060"),ylab = "Number of records (%)", xlab = "Price class",main = "Distribution of price class after sampling",names.arg = c("Reasonable", "Overpriced"))
  text(bp, 0,round(tbl,1),cex=1,pos=3)
  
  model_input <- balanced_data

  
  #Randomly reorder the data
  
  model_input <-  model_input[sample(1:nrow(model_input)),]
  
  
  return(model_input)
}


# ************************************************
# logistic_regression()
#
# INPUT: Data Frame
#
# OUTPUT : Generate logistic regression model and plots the required graphs
#         Also calculates model metrics
# ************************************************
logistic_regression <- function(training_data,testing_data)
{
  
  # Create a logistic regression model
  

  
  #Changing price_class to a factor variable
  
  
  #scatterPlotNonLinearRegression(training_data,testing_data,'numberofreviews','reviewspermonth',polyOrder =4)
  
  log_model <- glm(price_class ~ .,data=training_data, family ='binomial')
  print("*********************Models****************************")
  print("Summary of Logistic Regression model")
  #print(summary(log_model))

  # Prediction using the model
  suppressWarnings( log_predict <- predict(log_model,testing_data,type='response'))


  #Calculate and plot importance
  imp <- varImp(log_model, scale = TRUE)
  imp<-imp[order(imp,decreasing=TRUE),,drop=FALSE]
  print("Feature Importance")
  print(imp)
  par(mar = c(4,8,0,0))
  suppressWarnings( barplot(unlist(imp),names.arg =rownames(imp),las =2,horiz = TRUE,xlab = "Percentage of importance"))
  par(pty = "s")
  par(mar = c(5,3,3,3))
  
  
  measures <-determineThreshold(test_predicted = log_predict,test_expected = testing_data$price_class,plot = TRUE,title="Logistic Classifer Model")
  
  pred_direction <- rep(0, length(log_predict))
  pred_direction[log_predict >= measures$threshold] <- 1
  testing_data$price_class <- as.factor(testing_data$price_class)
  pred_direction <- as.factor(pred_direction)
  
  #Create confusion matrix
  print(confusionMatrix(testing_data$price_class,pred_direction))
  
  tbl <- table(pred_direction,testing_data$price_class)
  TP <- as.double(tbl[2,2])
  TN <- as.double(tbl[1,1])
  FN <- as.double(tbl[1,2])
  FP <- as.double(tbl[2,1])
  
  get_accuracy(TP,TN,FP,FN)
  classification_error_rate(TP,TN,FP,FN)
  precision <-get_precision(TP,TN,FP,FN)
  sensitivity <-get_sensitivity(TP,TN,FP,FN)
  specificity <- get_specificity(TP,TN,FP,FN)
  Calculate_f_score(precision,sensitivity)
  calculate_MCC(TP,TN,FP,FN)
  print("********************************************************")
  
}


# ************************************************
# decision_tree()
#
# INPUT: Data Frame, number of trials/boost
#
# OUTPUT : Evaluate decision tree model and plots the required graphs
#         Also calculates model metrics
# ************************************************
eval_decision_tree <- function(DT_mod,testing_data,heading,boost)
{
  PDF_FILENAME      <- "tree.pdf"           # Name of PDF with graphical tree diagram
  NODE_LEVEL        <- 1                    # The number is the node level of the tree to plot
  
  # Create a TRAINING dataset using first HOLDOUT% of the records
  # and the remaining 30% is used as TEST
  # use ALL fields (columns)
  # training_records<-round(nrow(model_input)*(HOLDOUT/100))
  # training_data <- model_input[1:training_records,]
  # testing_data = model_input[-(1:training_records),]
  # 
  
  #simpleDT(train = training_data, test = testing_data,plot = TRUE,k)
  
  #training_data_input <- subset(training_data, select=-c(price_class))
  #training_data_output <- factor(training_data$price_class)
  #DT_mod <- C50::C5.0(training_data_input,training_data_output,rules = FALSE,trials = boost)

  suppressWarnings(graphtree<-C50:::as.party.C5.0(DT_mod))

  # The plot is large - so print to a big PDF file
  pdf(PDF_FILENAME, width=100, height=50, paper="special", onefile=F)

  # The number is the node level of the tree to print
  plot(graphtree[NODE_LEVEL])

  #This closes the PDF file
  dev.off()


  print(paste("Summary of Decision Tree Boost =",boost))
 # print(summary(DT_mod))

  measures<-getTreeClassifications(myTree = DT_mod,
                                   testDataset = testing_data,
                                   title=heading)

  mod_predict <- predict(DT_mod,testing_data,type = "prob")
  pred_direction <- rep(0, nrow(mod_predict))
  pred_direction[mod_predict[,"1"] >= measures$threshold] <- 1
  testing_data$price_class <- as.factor(testing_data$price_class)
  pred_direction <- as.factor(pred_direction)

  #Create confusion matrix
  print(confusionMatrix(testing_data$price_class,pred_direction))

  tbl <- table(pred_direction,testing_data$price_class)
  TP <- as.double(tbl[2,2])
  TN <- as.double(tbl[1,1])
  FN <- as.double(tbl[1,2])
  FP <- as.double(tbl[2,1])

  get_accuracy(TP,TN,FP,FN)
  classification_error_rate(TP,TN,FP,FN)
  precision <-get_precision(TP,TN,FP,FN)
  sensitivity <-get_sensitivity(TP,TN,FP,FN)
  specificity <- get_specificity(TP,TN,FP,FN)
  Calculate_f_score(precision,sensitivity)
  calculate_MCC(TP,TN,FP,FN)


  imp <- varImp(DT_mod, scale = TRUE)
  imp<-imp[order(imp,decreasing=TRUE),,drop=FALSE]
  print("Importance of each fields")
  print(imp)
  suppressWarnings( barplot(unlist(imp),names.arg =rownames(imp),las =2,horiz = TRUE,xlab = "Percentage of importance"))
  print("***********************************************************************")
}


# ************************************************
# randomForest_model()
#
# INPUT: Training data, testing data, nTree, mtry
#
# OUTPUT : Evaluate random forest model and plots the required graphs
#         Also calculates model metrics
# ************************************************
randomForest_model <- function(training_data.testing_data,numtree,m)
{
  # train data: dataframe with the input fields
  train_inputs<-subset (training_data, select = -c(price_class))
  
  # train data: vector with the expedcted output
  train_expected<-training_data$price_class
  
  
  rf<-randomForest::randomForest(train_inputs,
                                 factor(train_expected),
                                 ntree=numtree ,
                                 importance=TRUE,
                                 mtry=m)
  
  
  # ************************************************
  # Use the created decision tree with the test dataset
  measures<-getTreeClassifications(myTree = rf,
                                   testDataset = testing_data,
                                   title="RF",
                                   plot=TRUE,
  )
  mod_predict <- predict(rf,testing_data)
  print("Summary of Random forest model")
  #print(summary(rf))
  

  imp <- randomForest::importance((rf))
  imp <- as.data.frame(imp)
  
  imp <- cbind(vars = rownames(imp),imp)
  imp <- subset(imp, select= c(vars,MeanDecreaseAccuracy))
  imp <- imp[with(imp,order(MeanDecreaseAccuracy)),]
  barplot(imp$MeanDecreaseAccuracy,names.arg = imp$vars,xlab = "Importance", las = 2,horiz = TRUE )
  par(pty = "s")
  par(mar = c(5,3,3,3))
  heading <- paste("Random Forest - ntrees =",numtree)
  measures<-getTreeClassifications(myTree = rf,
                                   testDataset = testing_data,
                                   title=heading)
  
  mod_predict <- predict(rf,testing_data,type = "prob")
  pred_direction <- rep(0, nrow(mod_predict))
  pred_direction[mod_predict[,"1"] >= measures$threshold] <- 1
  testing_data$price_class <- as.factor(testing_data$price_class)
  pred_direction <- as.factor(pred_direction)
  
  #Create confusion matrix
  print(confusionMatrix(testing_data$price_class,pred_direction))
  
  tbl <- table(pred_direction,testing_data$price_class)
  TP <- as.double(tbl[2,2])
  TN <- as.double(tbl[1,1])
  FN <- as.double(tbl[1,2])
  FP <- as.double(tbl[2,1])
  
  get_accuracy(TP,TN,FP,FN)
  classification_error_rate(TP,TN,FP,FN)
  precision <-get_precision(TP,TN,FP,FN)
  sensitivity <-get_sensitivity(TP,TN,FP,FN)
  specificity <- get_specificity(TP,TN,FP,FN)
  Calculate_f_score(precision,sensitivity)
  calculate_MCC(TP,TN,FP,FN)

  print("***********************************************************************")
}


# ************************************************
# NN_model()
#
# INPUT: Training data, testing data
#
# OUTPUT : Evaluate Neural network model and plots the required graphs
#         Also calculates model metrics
# ************************************************
NN_model <-function(training_data,testing_data)
{
  backup <- testing_data   #For auroc
  
  #h2o frame requires factor inputs
  testing_data$price_class <- as.factor(testing_data$price_class)
  training_data$price_class <- as.factor(training_data$price_class)
  
  #To create an h2o frame
  training_data_h2o <- as.h2o(training_data)
  testing_data_h2o <- as.h2o(testing_data)

  
  # Start an instance of h2o


  output_field_index <- which(colnames(model_input)=="price_class")
  input_field_index <- 1:(output_field_index-1)
  
  #Model object creation
  h2o_nn <- h2o.deeplearning(x = input_field_index,
                             y=output_field_index,
                             training_frame = training_data_h2o,
                             nfolds = 5,
                             #                          fold_assignment = "Stratified",
                             standardize = TRUE,
                             activation = "Rectifier",
                             hidden = c(20,20),
                             seed = "1234",
                             epochs = 30,
                             balance_classes = TRUE,
                             variable_importances = TRUE)
  h2o_predictions <- h2o.predict(h2o_nn,testing_data_h2o)
  
  h2o_predictions <- as.data.frame(h2o_predictions)
  par(pty = "s")
  par(mar = c(5,3,3,3))
  measures<-getClassifications(mod = h2o_nn,
                               testDataset = backup,
                               title="Deep Neural Network model")
  
  plot(h2o_nn,timestep = "AUTO", metrix = "AUTO")
  h2o.learning_curve_plot(h2o_nn,metric = "misclassification")
  print("Summary of DNN model")
  #print(summary(h2o_nn))
  pred_direction <- rep(0, nrow(h2o_predictions))
  pred_direction[h2o_predictions$p1 >= measures$threshold] <- 1
  
  
  testing <- as.data.frame(testing_data_h2o)
  testing$price_class <- as.factor(testing$price_class)
  pred_direction <- as.factor(pred_direction)
  
  #Create confusion matrix
  print(confusionMatrix(testing$price_class,pred_direction))
  
  tbl <- table(pred_direction,testing_data$price_class)
  TP <- as.double(tbl[2,2])
  TN <- as.double(tbl[1,1])
  FN <- as.double(tbl[1,2])
  FP <- as.double(tbl[2,1])
  
  get_accuracy(TP,TN,FP,FN)
  classification_error_rate(TP,TN,FP,FN)
  precision <-get_precision(TP,TN,FP,FN)
  sensitivity <-get_sensitivity(TP,TN,FP,FN)
  specificity <- get_specificity(TP,TN,FP,FN)
  Calculate_f_score(precision,sensitivity)
  calculate_MCC(TP,TN,FP,FN)
  
  
  h2o.varimp_plot(h2o_nn,num_of_features = 15)
  print("***********************************************************************")
  print("End of Execution")

}