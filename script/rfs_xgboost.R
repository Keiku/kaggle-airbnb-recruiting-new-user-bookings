labels <- readRDS(paste0("cache/",folder,"/labels.RData"))
sample_submission_NDF <- readRDS(paste0("cache/",folder,"/sample_submission_NDF.RData"))
df_all <- readRDS(paste0("cache/",folder,"/df_all.RData"))
df_all_feats <- readRDS(paste0("cache/",folder,"/df_all_feats.RData"))

X2_stack <- readRDS(paste0("cache/",folder,"/X2_stack.RData"))
X2_test_stack <- readRDS(paste0("cache/",folder,"/X2_test_stack.RData"))

X3_stack <- readRDS(paste0("cache/",folder,"/X3_stack.RData"))
X3_test_stack <- readRDS(paste0("cache/",folder,"/X3_test_stack.RData"))

X4_stack <- readRDS(paste0("cache/",folder,"/X4_stack.RData"))
X4_test_stack <- readRDS(paste0("cache/",folder,"/X4_test_stack.RData"))

X5_stack <- readRDS(paste0("cache/",folder,"/X5_stack.RData"))
X5_test_stack <- readRDS(paste0("cache/",folder,"/X5_test_stack.RData"))

X6_stack <- readRDS(paste0("cache/",folder,"/X6_stack.RData"))
X6_test_stack <- readRDS(paste0("cache/",folder,"/X6_test_stack.RData"))

X7_stack <- readRDS(paste0("cache/",folder,"/X7_stack.RData"))
X7_test_stack <- readRDS(paste0("cache/",folder,"/X7_test_stack.RData"))

X8_stack <- readRDS(paste0("cache/",folder,"/X8_stack.RData"))
X8_test_stack <- readRDS(paste0("cache/",folder,"/X8_test_stack.RData"))

X9_stack <- readRDS(paste0("cache/",folder,"/X9_stack.RData"))
X9_test_stack <- readRDS(paste0("cache/",folder,"/X9_test_stack.RData"))

# X10_stack <- readRDS(paste0("cache/",folder,"/X10_stack.RData"))
# X10_test_stack <- readRDS(paste0("cache/",folder,"/X10_test_stack.RData"))
# 
# X11_stack <- readRDS(paste0("cache/",folder,"/X11_stack.RData"))
# X11_test_stack <- readRDS(paste0("cache/",folder,"/X11_test_stack.RData"))

X12_stack <- readRDS(paste0("cache/",folder,"/X12_stack.RData"))
X12_test_stack <- readRDS(paste0("cache/",folder,"/X12_test_stack.RData"))

X13_stack <- readRDS(paste0("cache/",folder,"/X13_stack.RData"))
X13_test_stack <- readRDS(paste0("cache/",folder,"/X13_test_stack.RData"))

X14_stack <- readRDS(paste0("cache/",folder,"/X14_stack.RData"))
X14_test_stack <- readRDS(paste0("cache/",folder,"/X14_test_stack.RData"))

X15_stack <- readRDS(paste0("cache/",folder,"/X15_stack.RData"))
X15_test_stack <- readRDS(paste0("cache/",folder,"/X15_test_stack.RData"))

X16_stack <- readRDS(paste0("cache/",folder,"/X16_stack.RData"))
X16_test_stack <- readRDS(paste0("cache/",folder,"/X16_test_stack.RData"))

X17_stack <- readRDS(paste0("cache/",folder,"/X17_stack.RData"))
X17_test_stack <- readRDS(paste0("cache/",folder,"/X17_test_stack.RData"))

X18_stack <- readRDS(paste0("cache/",folder,"/X18_stack.RData"))
X18_test_stack <- readRDS(paste0("cache/",folder,"/X18_test_stack.RData"))

X19_stack <- readRDS(paste0("cache/",folder,"/X19_stack.RData"))
X19_test_stack <- readRDS(paste0("cache/",folder,"/X19_test_stack.RData"))

X_all <- subset(df_all_feats, is.na(value)==F)
X_all <- bind_rows(
  X_all,
  X2_stack,
  X2_test_stack,
  # X3_stack,
  # X3_test_stack,
  X4_stack,
  X4_test_stack,
  X5_stack,
  X5_test_stack,
  X6_stack,
  X6_test_stack,
  X7_stack,
  X7_test_stack,
  X8_stack,
  X8_test_stack,
  X9_stack,
  X9_test_stack,
  # X10_stack,
  # X10_test_stack,
  # X11_stack,
  # X11_test_stack
  X12_stack,
  X12_test_stack,
  X13_stack,
  X13_test_stack,
  X14_stack,
  X14_test_stack,
  X15_stack,
  X15_test_stack,
  X16_stack,
  X16_test_stack,
  X17_stack,
  X17_test_stack,
  X18_stack,
  X18_test_stack,
  X19_stack,
  X19_test_stack#,
)

X_all <- subset(X_all, id != "")
X_all$feature_name <- X_all$feature
X_all$feature <- as.numeric(as.factor(X_all$feature))

X_all_feature <- X_all[!duplicated(X_all$feature), c("feature", "feature_name")]
X_all_feature <- X_all_feature[order(X_all_feature$feature),]

X_all$id_num <- as.numeric(as.factor(X_all$id))

X_all_id <- X_all %>% distinct(id_num)
X_all_id <- data.frame(id = X_all_id$id, id_num = X_all_id$id_num)
X_all_id <- dplyr::left_join(X_all_id, labels, by = "id")
X_all_id <- X_all_id %>%
  dplyr::mutate(
    country_destination_num = recode(country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
  ) %>%
  dplyr::arrange(id_num)
y_all <- X_all_id$country_destination_num

X_all <- na.omit(X_all)
X_all_sp <- sparseMatrix(i = X_all$id_num,
                         j = X_all$feature,
                         x = X_all$value)
dim(X_all_sp)

X_all_dense <- data.frame(as.matrix(X_all_sp))
names(X_all_dense) <- X_all_feature$feature_name

X_dense <- subset(X_all_dense, dac_yearmonth %nin% c("201407", "201408", "201409"))
X_index <- as.integer(row.names(X_all_dense[X_all_dense$dac_yearmonth %nin% c("201407", "201408", "201409"), ]))
dim(X_dense)
X_train_dense <- subset(X_dense, dac_yearmonthweek %nin% c("20140521", "20140522", "20140622", "20140623", "20140624", "20140625", "20140626"))
X_train_index <- as.integer(row.names(X_dense[X_dense$dac_yearmonthweek %nin% c("20140521", "20140522", "20140622", "20140623", "20140624", "20140625", "20140626"), ]))
dim(X_train_dense)
X_valid_dense <- subset(X_dense, dac_yearmonthweek %in% c("20140521", "20140522", "20140622", "20140623", "20140624", "20140625", "20140626"))
X_valid_index <- as.integer(row.names(X_dense[X_dense$dac_yearmonthweek %in% c("20140521", "20140522", "20140622", "20140623", "20140624", "20140625", "20140626"), ]))
dim(X_valid_dense)
X_test_dense <- subset(X_all_dense, dac_yearmonth %in% c("201407", "201408", "201409"))
X_test_index <- as.integer(row.names(X_all_dense[X_all_dense$dac_yearmonth %in% c("201407", "201408", "201409"), ]))
dim(X_test_dense)

y <- y_all[X_index]
y_train <- y_all[X_train_index]
y_valid <- y_all[X_valid_index]
y_test <- y_all[X_test_index]

X_id <- X_all_id[X_index,]
X_train_id <- X_all_id[X_train_index,]
X_valid_id <- X_all_id[X_valid_index,]
X_test_id <- X_all_id[X_test_index,]

X_sp <- data.matrix(X_dense)
X_train_sp <- data.matrix(X_train_dense)
X_valid_sp <- data.matrix(X_valid_dense)
X_test_sp <- data.matrix(X_test_dense)

dX <- xgb.DMatrix(X_sp, label = y, missing = -99999)
dX_test <- xgb.DMatrix(X_test_sp, label = y_test, missing = -99999)

dX_train <- xgb.DMatrix(X_train_sp, label = y_train, missing = -99999)
dX_valid <- xgb.DMatrix(X_valid_sp, label = y_valid, missing = -99999)

saveRDS(X_all_feature, paste0("cache/",folder,"/X_all_feature.RData"))
saveRDS(X_id, paste0("cache/",folder,"/X_id.RData"))
saveRDS(X_test_id, paste0("cache/",folder,"/X_test_id.RData"))
saveRDS(X_train_id, paste0("cache/",folder,"/X_train_id.RData"))
saveRDS(X_valid_id, paste0("cache/",folder,"/X_valid_id.RData"))

saveRDS(y, paste0("cache/",folder,"/y.RData"))
saveRDS(y_test, paste0("cache/",folder,"/y_test.RData"))
saveRDS(y_train, paste0("cache/",folder,"/y_train.RData"))
saveRDS(y_valid, paste0("cache/",folder,"/y_valid.RData"))

saveRDS(X_sp, paste0("cache/",folder,"/X_sp.RData"))
saveRDS(X_test_sp, paste0("cache/",folder,"/X_test_sp.RData"))
saveRDS(X_train_sp, paste0("cache/",folder,"/X_train_sp.RData"))
saveRDS(X_valid_sp, paste0("cache/",folder,"/X_valid_sp.RData"))

xgb.DMatrix.save(dX, paste0("cache/",folder,"/dX.xgb.DMatrix.data"))
xgb.DMatrix.save(dX_test, paste0("cache/",folder,"/dX_test.xgb.DMatrix.data"))
xgb.DMatrix.save(dX_train, paste0("cache/",folder,"/dX_train.xgb.DMatrix.data"))
xgb.DMatrix.save(dX_valid, paste0("cache/",folder,"/dX_valid.xgb.DMatrix.data"))

X_all_feature <- readRDS(paste0("cache/",folder,"/X_all_feature.RData"))
X_id <- readRDS(paste0("cache/",folder,"/X_id.RData"))
X_test_id <- readRDS(paste0("cache/",folder,"/X_test_id.RData"))
X_train_id <- readRDS(paste0("cache/",folder,"/X_train_id.RData"))
X_valid_id <- readRDS(paste0("cache/",folder,"/X_valid_id.RData"))

y <- readRDS(paste0("cache/",folder,"/y.RData"))
y_test <- readRDS(paste0("cache/",folder,"/y_test.RData"))
y_train <- readRDS(paste0("cache/",folder,"/y_train.RData"))
y_valid <- readRDS(paste0("cache/",folder,"/y_valid.RData"))

X_sp <- readRDS(paste0("cache/",folder,"/X_sp.RData"))
X_test_sp <- readRDS(paste0("cache/",folder,"/X_test_sp.RData"))
X_train_sp <- readRDS(paste0("cache/",folder,"/X_train_sp.RData"))
X_valid_sp <- readRDS(paste0("cache/",folder,"/X_valid_sp.RData"))

dX <- xgb.DMatrix(paste0("cache/",folder,"/dX.xgb.DMatrix.data"))
dX_test <- xgb.DMatrix(paste0("cache/",folder,"/dX_test.xgb.DMatrix.data"))
dX_train <- xgb.DMatrix(paste0("cache/",folder,"/dX_train.xgb.DMatrix.data"))
dX_test <- xgb.DMatrix(paste0("cache/",folder,"/dX_test.xgb.DMatrix.data"))

# **************************************
# randomized feature selection
# **************************************

nDCG_score_df <- data.frame()
for(i in 1:90){
  # i <- 1
  start_time <- proc.time()
  set.seed(123 * i)
  rate <- 0.9
  
  feature_set_index <- createDataPartition(1:ncol(X_sp), p = rate,
                                           list = FALSE,
                                           times = 1)
  feature_set_index <- feature_set_index[,1]
  X_sp_ <- X_sp[, feature_set_index]
  X_test_sp_ <- X_test_sp[, feature_set_index]
  X_all_feature_ <- subset(X_all_feature, feature %in% feature_set_index)
  
  dX_ <- xgb.DMatrix(X_sp_, label = y, missing = -99999)
  
  param <- list("objective" = "multi:softprob",
                "num_class" = 12,
                # "eval_metric" = "merror",
                # "eval_metric" = "mlogloss", 
                "eta" = 0.05,
                "max_depth" = 6,
                "subsample" = 0.5,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  # Run Cross Valication
  # cv.nround = 2
  cv.nround = 3000
  bst.cv = xgb.cv(param = param,
                  data = dX_, 
                  nfold = 5,
                  nrounds = cv.nround,
                  maximize = TRUE,
                  feval = ndcg5,
                  early.stop.round = 10
  )

  elapse_time <- proc.time() - start_time
  nDCG_score_df_ <- data.frame(
    feature_set = i,
    nDCG = Max(bst.cv$test.ndcg5.mean),
    elapse_time = elapse_time[[3]],
    n = which.max(bst.cv$test.ndcg5.mean)
  )
  nDCG_score_df <- bind_rows(
    nDCG_score_df,
    nDCG_score_df_
  )
  saveRDS(X_all_feature_, paste0("cache/",folder,"/test/X_all_feature",i,".RData"))
  saveRDS(nDCG_score_df, paste0("cache/",folder,"/test/nDCG_score_df.RData"))
}


# **************************************
# randomized feature selection (part2)
# **************************************

nDCG_score_df <- readRDS(paste0("cache/",folder,"/test/nDCG_score_df.RData"))

i <- which.max(nDCG_score_df$nDCG)
n <- nDCG_score_df$n[which.max(nDCG_score_df$n)]
X_all_feature <- readRDS(paste0("cache/",folder,"/X_all_feature.RData"))
X_all_feature_ <- readRDS(paste0("cache/",folder,"/test/X_all_feature",i,".RData"))

X_all_feature_select <- subset(X_all_feature, feature_name %in% X_all_feature_$feature_name)

X_sp <- X_sp[, X_all_feature_select$feature]
X_test_sp <- X_test_sp[, X_all_feature_select$feature]

nDCG_score_df <- data.frame()
for(i in 1:90){# 
  # i <- 1
  start_time <- proc.time()
  set.seed(234 * i)
  rate <- 0.9
  
  feature_set_index <- createDataPartition(1:ncol(X_sp), p = rate,
                                           list = FALSE,
                                           times = 1)
  feature_set_index <- feature_set_index[,1]
  X_sp_ <- X_sp[, feature_set_index]
  X_test_sp_ <- X_test_sp[, feature_set_index]
  X_all_feature_ <- subset(X_all_feature, feature %in% feature_set_index)
  
  dX_ <- xgb.DMatrix(X_sp_, label = y, missing = -99999)
  
  param <- list("objective" = "multi:softprob",
                "num_class" = 12,
                # "eval_metric" = "merror",
                # "eval_metric" = "mlogloss", 
                "eta" = 0.05,
                "max_depth" = 6,
                "subsample" = 0.5,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  # Run Cross Valication
  # cv.nround = 2
  cv.nround = 3000
  bst.cv = xgb.cv(param = param,
                  data = dX_, 
                  nfold = 5,
                  nrounds=cv.nround,
                  maximize = TRUE,
                  feval = ndcg5,
                  early.stop.round = 10
  )
  
  if(Max(bst.cv$test.ndcg5.mean) > 0.833600){
    dX_test_ <- xgb.DMatrix(X_test_sp_, missing = -99999)
    # train xgboost
    xgb <- xgboost(data = dX_, 
                   eta = 0.05,
                   max_depth = 6, 
                   # nround = which.min(bst.cv$test.merror.mean), 
                   # nround = which.min(bst.cv$test.mlogloss.mean), 
                   nround = which.max(bst.cv$test.ndcg5.mean), 
                   # nround = n, 
                   subsample = 0.5,
                   colsample_bytree = 0.3,
                   alpha = 1.0,
                   # eval_metric = "merror",
                   # eval_metric = "mlogloss",
                   eval_metric = ndcg5,
                   prediction = TRUE,
                   maximize = TRUE,
                   objective = "multi:softprob",
                   num_class = 12,
                   nthread = 24
    )
    
    # **************************************
    # calculate XGBoost importance
    # **************************************
    # X_all_imp <- xgb.importance(X_all_feature$feature_name, model=xgb)
    # saveRDS(X_all_imp_, paste0("cache/",folder,"test/X_all_imp",i,".RData"))
    
    # predict values in test set
    y_pred <- predict(xgb, dX_test_)
    
    predictions <- as.data.frame(t(matrix(y_pred, nrow=12)))
    colnames(predictions) <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
    
    X_test_predictions <- bind_cols(
      X_test_id,
      predictions 
    )
    saveRDS(X_test_predictions, paste0("cache/",folder,"/test/X_test_predictions_part2_",i,".RData"))
  }
  
  elapse_time <- proc.time() - start_time
  nDCG_score_df_ <- data.frame(
    feature_set = i,
    nDCG = Max(bst.cv$test.ndcg5.mean),
    elapse_time = elapse_time[[3]],
    n = which.max(bst.cv$test.ndcg5.mean)
  )
  nDCG_score_df <- bind_rows(
    nDCG_score_df,
    nDCG_score_df_
  )
  saveRDS(X_all_feature_, paste0("cache/",folder,"/test/X_all_feature_part2_",i,".RData"))
  saveRDS(nDCG_score_df, paste0("cache/",folder,"/test/nDCG_score_df_part2.RData"))
}