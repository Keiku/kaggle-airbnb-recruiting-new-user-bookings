# **************************************
# options
# **************************************
options(scipen = 100)
options(dplyr.width = Inf)
options(dplyr.print_max = Inf)

# **************************************
# packages
# **************************************
# install package
install_packages <- 0
if(install_packages == 1){
  install.packages(c("Hmisc",
                     "xgboost",
                     "readr",
                     "stringr",
                     "caret",
                     "car",
                     "plyr",
                     "dplyr",
                     "tidyr",
                     "data.table",
                     "DescTools",
                     "Matrix",
                     "glmnet"))
}

# load libraries
library(Hmisc)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(plyr)
library(dplyr)
library(tidyr)
library(data.table)
library(DescTools)
library(Matrix)
library(glmnet)

# **************************************
# functions
# **************************************
'%nin%' <- Negate('%in%')
'%in_v%' <- function(x, y) x[x %in% y] 
'%nin_v%' <- function(x, y) x[!x %in% y] 
'%in_d%' <- function(x, y) x[names(x) %in% y] 
'%nin_d%' <- function(x, y) x[!names(x) %in% y] 
'%+%' <- function(x, y) paste0(x, y)

Mean <- function(x) mean(x, na.rm=TRUE)
Median <- function(x) median(x, na.rm=TRUE)
Sd <- function(x) sd(x, na.rm=TRUE)
Sum <- function(x) sum(x, na.rm=TRUE)
Max <- function(x) max(x, na.rm=TRUE)
Min <- function(x) min(x, na.rm=TRUE)
Mean_value <- function(x) ifelse(is.nan(mean(x, na.rm=TRUE))==T, NA, mean(x, na.rm=TRUE))
Sum_value <- function(x) ifelse(sum(!is.na(x))==0, NA, sum(x, na.rm=TRUE))
Max_value <- function(x) ifelse(is.infinite(max(x, na.rm=TRUE))==T, NA, max(x, na.rm=TRUE))
Min_value <- function(x) ifelse(is.infinite(min(x, na.rm=TRUE))==T, NA, min(x, na.rm=TRUE))

freq <- function(df, remove0, ...) {
  input_list <- list(...)
  df_list <- lapply(X=input_list, function(x) {df[[x]]})
  df <- as.data.frame(table(df_list, useNA = "always"))
  names(df) <- c(unlist(input_list), "Freq")
  df$Percent <- df$Freq / sum(df$Freq)
  for(var in rev(input_list)){ df <- df[order(df[[var]]),] }
  df$CumSum <- cumsum(df$Freq) 
  df$CumPercent <- df$CumSum / sum(df$Freq)
  if(remove0==TRUE){ df <- subset(df, Freq!=0) }
  df
}

# **************************************
# ndcg5 metric
# **************************************
dcg_at_k <- function (r, k=min(5, length(r)) ) {
  #only coded alternative formulation of DCG (used by kaggle)
  r <- as.vector(r)[1:k]
  sum(( 2^r - 1 )/ log2( 2:(length(r)+1)) )
} 

ndcg_at_k <- function(r, k=min(5, length(r)) ) {
  r <- as.vector(r)[1:k]
  if (sum(r) <= 0) return (0)     # no hits (dcg_max = 0)
  dcg_max = dcg_at_k(sort(r, decreasing=TRUE)[1:k], k)
  return ( dcg_at_k(r, k) / dcg_max )
}

score_predictions <- function(preds, truth) {
  # preds: matrix or data.frame
  # one row for each observation, one column for each prediction.
  # Columns are sorted from left to right descending in order of likelihood.
  # truth: vector
  # one row for each observation.
  preds <- as.matrix(preds)
  truth <- as.vector(truth)
  
  stopifnot( length(truth) == nrow(preds))
  r <- apply( cbind( truth, preds), 1
              , function(x) ifelse( x == x[1], 1, 0))[ -1, ]
  if ( ncol(preds) == 1) r <-  rbind( r, r)  #workaround for 1d matrices
  as.vector( apply(r, 2, ndcg_at_k) )
}

ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = length(unique(labels))
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}