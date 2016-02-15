# **************************************
# stacking in the following model.
# **************************************
# X2: 
# target: dfb_dac_lag_flg
# model: XGBoost
# training-dataset: non-missing part of dfb_dac_lag_flg
# **************************************
# X3: 
# target: age
# model: XGBoost
# training-dataset: non-missing part of age
# **************************************
# X4: 
# target: country_destination
# model: XGBoost
# training-dataset: all training dataset
# **************************************
# X5:
# target: country_destination
# model: XGBoost
# training-dataset: last 12 months of all training dataset
# **************************************
# X6:
# target: country_destination
# model: XGBoost
# training-dataset: last 6 months of all training dataset
# **************************************
# X7:
# target: country_destination
# model: XGBoost
# training-dataset: last summer of all training dataset
# **************************************
# X8:
# target: age_cln
# model: XGBoost
# training-dataset: non-missing part of age_cln
# **************************************
# X9:
# target: age_cln2
# model: XGBoost
# training-dataset: non-missing part of age_cln2
# **************************************
# X10:
# target: distance_km
# model: XGBoost
# training-dataset: non-missing part of distance_km
# **************************************
# X11:
# target: destination_km2
# model: XGBoost
# training-dataset: non-missing part of destination_km2
# **************************************
# X12: gender
# target: gender
# model: XGBoost
# training-dataset: already known gender part
# **************************************
# X13:
# target: dfb_tfa_lag_flg
# model: XGBoost
# training-dataset: non-missing part of dfb_tfa_lag_flg
# **************************************
# X14: dfb_dac_lag
# target: dfb_dac_lag
# model: XGBoost
# training-dataset: non-missing part of dfb_dac_lag
# **************************************
# X15: dfb_tfa_lag
# target: dfb_tfa_lag
# model: XGBoost
# training-dataset: non-missing part of dfb_tfa_lag
# **************************************
# X16:
# target: age_cln
# model: glmnet
# training-dataset: non-missing part of age_cln
# **************************************
# X17:
# target: age_cln2
# model: glmnet
# training-dataset: non-missing part of age_cln2
# **************************************
# X18:
# target: dfb_dac_lag
# model: glmnet
# training-dataset: non-missing part of dfb_dac_lag
# **************************************
# X19:
# target: dfb_tfa_lag
# model: glmnet
# training-dataset: non-missing part of dfb_tfa_lag
# **************************************


# **************************************
# make dataset for predict dfb_dac_lag
# **************************************
X2_all <- subset(df_all_feats, is.na(value)==F)

X2_all$feature_name <- X2_all$feature
X2_all$feature <- as.numeric(as.factor(X2_all$feature))
X2 <- subset(X2_all, id %in% subset(df_all, dac_yearmonth %nin% c("201407", "201408", "201409"))$id)
X2_test <- subset(X2_all, id %in% subset(df_all, dac_yearmonth %in% c("201407", "201408", "201409"))$id)
X2_all <- rbind(X2, X2_test)

X2_all_feature <- X2_all[!duplicated(X2_all$feature), c("feature", "feature_name")]
X2_all_feature <- X2_all_feature[order(X2_all_feature$feature),]

X2$id_num <- as.numeric(as.factor(X2$id))
X2_test$id_num <- as.numeric(as.factor(X2_test$id))

X2_id <- X2 %>% distinct(id_num)
X2_id <- data.frame(id = X2_id$id, id_num = X2_id$id_num)
X2_id <- dplyr::left_join(X2_id, df_all[c("id", "dfb_dac_lag_flg")], by = "id")
X2_id <- X2_id %>%
  dplyr::arrange(id_num)
y2 <- X2_id$dfb_dac_lag_flg

X2_test_id <- X2_test %>% distinct(id_num)
X2_test_id <- data.frame(id = X2_test_id$id, id_num = X2_test_id$id_num)
X2_test_id <- X2_test_id %>%
  dplyr::arrange(id_num)
y2_test <- rep(NA, nrow(X2_test_id))


common_feature <- dplyr::intersect(unique(X2$feature), unique(X2_test$feature))

X2 <- na.omit(X2)
X2 <- subset(X2, feature %in% common_feature)
X2_sp <- sparseMatrix(i = X2$id_num,
                      j = X2$feature,
                      x = X2$value)
dim(X2_sp)

X2_test <- na.omit(X2_test)
X2_test <- subset(X2_test, feature %in% common_feature)
X2_test_sp <- sparseMatrix(i = X2_test$id_num,
                           j = X2_test$feature,
                           x = X2_test$value)
dim(X2_test_sp)

dX2 <- xgb.DMatrix(X2_sp, label = y2, missing = -99999)
dX2_test <- xgb.DMatrix(X2_test_sp, missing = -99999)

# **************************************
# Predict dfb_dac_lag_flg
# **************************************
Folds <- 10
X2_cv <- createFolds(1:nrow(X2_sp), k = Folds)
X2_stack <- data.frame()
X2_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X2_id_ <- X2_id[-X2_cv[[i]], ]
  X2_fold_id_ <- X2_id[X2_cv[[i]], ]
  X2_sp_ <- X2_sp[-X2_cv[[i]], ]
  X2_fold_sp_ <- X2_sp[X2_cv[[i]], ]
  y2_ <- y2[-X2_cv[[i]]]
  # y2_fold_ <- y2[X2_cv[[i]]]
  
  dX2_ <- xgb.DMatrix(X2_sp_, label = y2_)
  dX2_fold_ <- xgb.DMatrix(X2_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = 4,
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param = param,
                    data = dX2_, 
                    nfold = Folds,
                    nrounds = cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X2_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX2_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y2_fold_ <- predict(xgb, dX2_fold_)
  y2_fold_mat <- matrix(y2_fold_, nrow=nrow(X2_fold_sp_), ncol=n_distinct(y2_), byrow=T)
  y2_fold_df <- as.data.frame(y2_fold_mat)
  names(y2_fold_df) <- paste0("dfb_dac_lag_flg_", 1:4)
  y2_fold_df <- mutate(y2_fold_df,
                       id = unique(X2_fold_id_$id))
  y2_fold_df <- y2_fold_df[c("id", paste0("dfb_dac_lag_flg_", 1:4))]
  X2_stack <- bind_rows(X2_stack,
                        y2_fold_df)
  
  y2_test_ <- predict(xgb, dX2_test)
  y2_test_mat <- matrix(y2_test_, nrow=nrow(X2_test_sp), ncol=n_distinct(y2_), byrow=T)
  y2_test_df <- as.data.frame(y2_test_mat)
  names(y2_test_df) <- paste0("dfb_dac_lag_flg_", 1:4)
  y2_test_df <- mutate(y2_test_df,
                       id = unique(X2_test_id$id))
  y2_test_df <- y2_test_df[c("id", paste0("dfb_dac_lag_flg_", 1:4))]
  X2_test_stack <- bind_rows(X2_test_stack,
                             y2_test_df)
}
X2_test_stack <- X2_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X2_stack <- melt.data.table(as.data.table(X2_stack))
X2_stack <- data.frame(X2_stack)
names(X2_stack) <- c("id", "feature", "value")

X2_test_stack <- melt.data.table(as.data.table(X2_test_stack))
X2_test_stack <- data.frame(X2_test_stack)
names(X2_test_stack) <- c("id", "feature", "value")

saveRDS(X2_stack, paste0("cache/",folder,"/X2_stack.RData"))
saveRDS(X2_test_stack, paste0("cache/",folder,"/X2_test_stack.RData"))
gc()

# **************************************
# make dataset for predict age
# **************************************
X3_all <- subset(df_all_feats, is.na(value)==F)

X3_all <- subset(X3_all, feature %nin% c("age"))
X3_all <- subset(X3_all, feature %nin% c("age_cln"))
X3_all <- subset(X3_all, feature %nin% c("age_cln2"))

X3_all$feature_name <- X3_all$feature
X3_all$feature <- as.numeric(as.factor(X3_all$feature))
X3 <- subset(X3_all, id %in% subset(df_all, is.na(age)==F)$id)
X3_test <- subset(X3_all, id %in% subset(df_all, is.na(age)==T)$id)
X3_all <- rbind(X3, X3_test)

X3_all_feature <- X3_all[!duplicated(X3_all$feature), c("feature", "feature_name")]
X3_all_feature <- X3_all_feature[order(X3_all_feature$feature),]

X3$id_num <- as.numeric(as.factor(X3$id))
X3_test$id_num <- as.numeric(as.factor(X3_test$id))

X3_id <- X3 %>% distinct(id_num)
X3_id <- data.frame(id = X3_id$id, id_num = X3_id$id_num)
X3_id <- dplyr::left_join(X3_id, df_all[c("id", "age")], by = "id")
X3_id <- X3_id %>%
  dplyr::arrange(id_num)
y3 <- X3_id$age

X3_test_id <- X3_test %>% distinct(id_num)
X3_test_id <- data.frame(id = X3_test_id$id, id_num = X3_test_id$id_num)
X3_test_id <- X3_test_id %>%
  dplyr::arrange(id_num)
y3_test <- rep(NA, nrow(X3_test_id))


common_feature <- dplyr::intersect(unique(X3$feature), unique(X3_test$feature))

X3 <- na.omit(X3)
X3 <- subset(X3, feature %in% common_feature)
X3_sp <- sparseMatrix(i = X3$id_num,
                      j = X3$feature,
                      x = X3$value)
dim(X3_sp)

X3_test <- na.omit(X3_test)
X3_test <- subset(X3_test, feature %in% common_feature)
X3_test_sp <- sparseMatrix(i = X3_test$id_num,
                           j = X3_test$feature,
                           x = X3_test$value)
dim(X3_test_sp)

dX3 <- xgb.DMatrix(X3_sp, label = y3, missing = -99999)
dX3_test <- xgb.DMatrix(X3_test_sp, missing = -99999)

# **************************************
# Predict age
# **************************************
Folds <- 10
X3_cv <- createFolds(1:nrow(X3_sp), k = Folds)
X3_stack <- data.frame()
X3_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X3_id_ <- X3_id[-X3_cv[[i]], ]
  X3_fold_id_ <- X3_id[X3_cv[[i]], ]
  X3_sp_ <- X3_sp[-X3_cv[[i]], ]
  X3_fold_sp_ <- X3_sp[X3_cv[[i]], ]
  y3_ <- y3[-X3_cv[[i]]]
  # y3_fold_ <- y3[X3_cv[[i]]]
  
  dX3_ <- xgb.DMatrix(X3_sp_, label = y3_)
  dX3_fold_ <- xgb.DMatrix(X3_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX3_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X3_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX3_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y3_fold_ <- predict(xgb, dX3_fold_)
  y3_fold_df <- data.frame(age_pred = y3_fold_)
  y3_fold_df <- mutate(y3_fold_df,
                       id = unique(X3_fold_id_$id))
  y3_fold_df <- y3_fold_df[c("id", "age_pred")]
  X3_stack <- bind_rows(X3_stack,
                        y3_fold_df)
  
  y3_test_ <- predict(xgb, dX3_test)
  y3_test_df <- data.frame(age_pred = y3_test_)
  y3_test_df <- mutate(y3_test_df,
                       id = unique(X3_test_id$id))
  y3_test_df <- y3_test_df[c("id", "age_pred")]
  X3_test_stack <- bind_rows(X3_test_stack,
                             y3_test_df)
}
X3_test_stack <- X3_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X3_stack <- melt.data.table(as.data.table(X3_stack))
X3_stack <- data.frame(X3_stack)
names(X3_stack) <- c("id", "feature", "value")

X3_test_stack <- melt.data.table(as.data.table(X3_test_stack))
X3_test_stack <- data.frame(X3_test_stack)
names(X3_test_stack) <- c("id", "feature", "value")

saveRDS(X3_stack, paste0("cache/",folder,"/X3_stack.RData"))
saveRDS(X3_test_stack, paste0("cache/",folder,"/X3_test_stack.RData"))
gc()

# **************************************
# make dataset for predict country_destination
# **************************************
X4_all <- subset(df_all_feats, is.na(value)==F)

X4_all$feature_name <- X4_all$feature
X4_all$feature <- as.numeric(as.factor(X4_all$feature))
X4 <- subset(X4_all, id %in% subset(df_all, dac_yearmonth %nin% c("201407", "201408", "201409"))$id)
X4_test <- subset(X4_all, id %in% subset(df_all, dac_yearmonth %in% c("201407", "201408", "201409"))$id)
X4_all <- rbind(X4, X4_test)

X4_all_feature <- X4_all[!duplicated(X4_all$feature), c("feature", "feature_name")]
X4_all_feature <- X4_all_feature[order(X4_all_feature$feature),]

X4$id_num <- as.numeric(as.factor(X4$id))
X4_test$id_num <- as.numeric(as.factor(X4_test$id))

X4_id <- X4 %>% distinct(id_num)
X4_id <- data.frame(id = X4_id$id, id_num = X4_id$id_num)
X4_id <- dplyr::left_join(X4_id, labels, by = "id")
X4_id <- X4_id %>%
  dplyr::mutate(
    country_destination_num = recode(country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
  ) %>%
  dplyr::arrange(id_num)
y4 <- X4_id$country_destination_num

X4_test_id <- X4_test %>% distinct(id_num)
X4_test_id <- data.frame(id = X4_test_id$id, id_num = X4_test_id$id_num)
X4_test_id <- X4_test_id %>%
  dplyr::arrange(id_num)
y4_test <- rep(NA, nrow(X4_test_id))


common_feature <- dplyr::intersect(unique(X4$feature), unique(X4_test$feature))

X4 <- na.omit(X4)
X4 <- subset(X4, feature %in% common_feature)
X4_sp <- sparseMatrix(i = X4$id_num,
                      j = X4$feature,
                      x = X4$value)
dim(X4_sp)

X4_test <- na.omit(X4_test)
X4_test <- subset(X4_test, feature %in% common_feature)
X4_test_sp <- sparseMatrix(i = X4_test$id_num,
                           j = X4_test$feature,
                           x = X4_test$value)
dim(X4_test_sp)

dX4 <- xgb.DMatrix(X4_sp, label = y4, missing = -99999)
dX4_test <- xgb.DMatrix(X4_test_sp, missing = -99999)

# **************************************
# Predict country_destination
# **************************************
Folds <- 10
X4_cv <- createFolds(1:nrow(X4_sp), k = Folds)
X4_stack <- data.frame()
X4_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X4_id_ <- X4_id[-X4_cv[[i]], ]
  X4_fold_id_ <- X4_id[X4_cv[[i]], ]
  X4_sp_ <- X4_sp[-X4_cv[[i]], ]
  X4_fold_sp_ <- X4_sp[X4_cv[[i]], ]
  y4_ <- y4[-X4_cv[[i]]]
  # y4_fold_ <- y4[X4_cv[[i]]]
  
  dX4_ <- xgb.DMatrix(X4_sp_, label = y4_)
  dX4_fold_ <- xgb.DMatrix(X4_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y4),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param = param,
                    data = dX4_, 
                    nfold = Folds,
                    nrounds = cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X4_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX4_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y4_fold_ <- predict(xgb, dX4_fold_)
  y4_fold_mat <- matrix(y4_fold_, nrow=nrow(X4_fold_sp_), ncol=n_distinct(y4_), byrow=T)
  y4_fold_df <- as.data.frame(y4_fold_mat)
  country_destination_label <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
  names(y4_fold_df) <- paste0("pred1_", country_destination_label)
  y4_fold_df <- mutate(y4_fold_df,
                       id = unique(X4_fold_id_$id))
  y4_fold_df <- y4_fold_df[c("id", paste0("pred1_", country_destination_label))]
  X4_stack <- bind_rows(X4_stack,
                        y4_fold_df)
  
  y4_test_ <- predict(xgb, dX4_test)
  y4_test_mat <- matrix(y4_test_, nrow=nrow(X4_test_sp), ncol=n_distinct(y4_), byrow=T)
  y4_test_df <- as.data.frame(y4_test_mat)
  names(y4_test_df) <- paste0("pred1_", country_destination_label)
  y4_test_df <- mutate(y4_test_df,
                       id = unique(X4_test_id$id))
  y4_test_df <- y4_test_df[c("id", paste0("pred1_", country_destination_label))]
  X4_test_stack <- bind_rows(X4_test_stack,
                             y4_test_df)
}
X4_test_stack <- X4_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X4_stack <- melt.data.table(as.data.table(X4_stack))
X4_stack <- data.frame(X4_stack)
names(X4_stack) <- c("id", "feature", "value")

X4_test_stack <- melt.data.table(as.data.table(X4_test_stack))
X4_test_stack <- data.frame(X4_test_stack)
names(X4_test_stack) <- c("id", "feature", "value")

saveRDS(X4_stack, paste0("cache/",folder,"/X4_stack.RData"))
saveRDS(X4_test_stack, paste0("cache/",folder,"/X4_test_stack.RData"))
gc()

# **************************************
# make dataset for predict country_destination
# **************************************
X5_all <- subset(df_all_feats, is.na(value)==F)

X5_all$feature_name <- X5_all$feature
X5_all$feature <- as.numeric(as.factor(X5_all$feature))
X5 <- subset(X5_all, id %in% subset(df_all, dac_yearmonth %in% c("201307", "201308", "201309",
                                                                 "201310", "201311", "201312",
                                                                 "201401", "201402", "201403",
                                                                 "201404", "201405", "201406"))$id)
X5_test <- subset(X5_all, id %in% subset(df_all, dac_yearmonth %nin% c("201307", "201308", "201309",
                                                                       "201310", "201311", "201312",
                                                                       "201401", "201402", "201403",
                                                                       "201404", "201405", "201406"))$id)
X5_all <- rbind(X5, X5_test)

X5_all_feature <- X5_all[!duplicated(X5_all$feature), c("feature", "feature_name")]
X5_all_feature <- X5_all_feature[order(X5_all_feature$feature),]

X5$id_num <- as.numeric(as.factor(X5$id))
X5_test$id_num <- as.numeric(as.factor(X5_test$id))

X5_id <- X5 %>% distinct(id_num)
X5_id <- data.frame(id = X5_id$id, id_num = X5_id$id_num)
X5_id <- dplyr::left_join(X5_id, labels, by = "id")
X5_id <- X5_id %>%
  dplyr::mutate(
    country_destination_num = recode(country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
  ) %>%
  dplyr::arrange(id_num)
y5 <- X5_id$country_destination_num

X5_test_id <- X5_test %>% distinct(id_num)
X5_test_id <- data.frame(id = X5_test_id$id, id_num = X5_test_id$id_num)
X5_test_id <- X5_test_id %>%
  dplyr::arrange(id_num)
y5_test <- rep(NA, nrow(X5_test_id))


common_feature <- dplyr::intersect(unique(X5$feature), unique(X5_test$feature))

X5 <- na.omit(X5)
X5 <- subset(X5, feature %in% common_feature)
X5_sp <- sparseMatrix(i = X5$id_num,
                      j = X5$feature,
                      x = X5$value)
dim(X5_sp)

X5_test <- na.omit(X5_test)
X5_test <- subset(X5_test, feature %in% common_feature)
X5_test_sp <- sparseMatrix(i = X5_test$id_num,
                           j = X5_test$feature,
                           x = X5_test$value)
dim(X5_test_sp)

dX5 <- xgb.DMatrix(X5_sp, label = y5, missing = -99999)
dX5_test <- xgb.DMatrix(X5_test_sp, missing = -99999)

# **************************************
# Predict country_destination
# **************************************
Folds <- 10
X5_cv <- createFolds(1:nrow(X5_sp), k = Folds)
X5_stack <- data.frame()
X5_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X5_id_ <- X5_id[-X5_cv[[i]], ]
  X5_fold_id_ <- X5_id[X5_cv[[i]], ]
  X5_sp_ <- X5_sp[-X5_cv[[i]], ]
  X5_fold_sp_ <- X5_sp[X5_cv[[i]], ]
  y5_ <- y5[-X5_cv[[i]]]
  # y5_fold_ <- y5[X5_cv[[i]]]
  
  dX5_ <- xgb.DMatrix(X5_sp_, label = y5_)
  dX5_fold_ <- xgb.DMatrix(X5_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y5),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param = param,
                    data = dX5_, 
                    nfold = Folds,
                    nrounds = cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X5_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX5_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y5_fold_ <- predict(xgb, dX5_fold_)
  y5_fold_mat <- matrix(y5_fold_, nrow=nrow(X5_fold_sp_), ncol=n_distinct(y5_), byrow=T)
  y5_fold_df <- as.data.frame(y5_fold_mat)
  country_destination_label <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
  names(y5_fold_df) <- paste0("pred2_", country_destination_label)
  y5_fold_df <- mutate(y5_fold_df,
                       id = unique(X5_fold_id_$id))
  y5_fold_df <- y5_fold_df[c("id", paste0("pred2_", country_destination_label))]
  X5_stack <- bind_rows(X5_stack,
                        y5_fold_df)
  
  y5_test_ <- predict(xgb, dX5_test)
  y5_test_mat <- matrix(y5_test_, nrow=nrow(X5_test_sp), ncol=n_distinct(y5_), byrow=T)
  y5_test_df <- as.data.frame(y5_test_mat)
  names(y5_test_df) <- paste0("pred2_", country_destination_label)
  y5_test_df <- mutate(y5_test_df,
                       id = unique(X5_test_id$id))
  y5_test_df <- y5_test_df[c("id", paste0("pred2_", country_destination_label))]
  X5_test_stack <- bind_rows(X5_test_stack,
                             y5_test_df)
}
X5_test_stack <- X5_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X5_stack <- melt.data.table(as.data.table(X5_stack))
X5_stack <- data.frame(X5_stack)
names(X5_stack) <- c("id", "feature", "value")

X5_test_stack <- melt.data.table(as.data.table(X5_test_stack))
X5_test_stack <- data.frame(X5_test_stack)
names(X5_test_stack) <- c("id", "feature", "value")

saveRDS(X5_stack, paste0("cache/",folder,"/X5_stack.RData"))
saveRDS(X5_test_stack, paste0("cache/",folder,"/X5_test_stack.RData"))
gc()

# **************************************
# make dataset for predict country_destination
# **************************************
X6_all <- subset(df_all_feats, is.na(value)==F)

X6_all$feature_name <- X6_all$feature
X6_all$feature <- as.numeric(as.factor(X6_all$feature))
X6 <- subset(X6_all, id %in% subset(df_all, dac_yearmonth %in% c("201401", "201402", "201403",
                                                                 "201404", "201405", "201406"))$id)
X6_test <- subset(X6_all, id %in% subset(df_all, dac_yearmonth %nin% c("201401", "201402", "201403",
                                                                       "201404", "201405", "201406"))$id)
X6_all <- rbind(X6, X6_test)

X6_all_feature <- X6_all[!duplicated(X6_all$feature), c("feature", "feature_name")]
X6_all_feature <- X6_all_feature[order(X6_all_feature$feature),]

X6$id_num <- as.numeric(as.factor(X6$id))
X6_test$id_num <- as.numeric(as.factor(X6_test$id))

X6_id <- X6 %>% distinct(id_num)
X6_id <- data.frame(id = X6_id$id, id_num = X6_id$id_num)
X6_id <- dplyr::left_join(X6_id, labels, by = "id")
X6_id <- X6_id %>%
  dplyr::mutate(
    country_destination_num = recode(country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
  ) %>%
  dplyr::arrange(id_num)
y6 <- X6_id$country_destination_num

X6_test_id <- X6_test %>% distinct(id_num)
X6_test_id <- data.frame(id = X6_test_id$id, id_num = X6_test_id$id_num)
X6_test_id <- X6_test_id %>%
  dplyr::arrange(id_num)
y6_test <- rep(NA, nrow(X6_test_id))


common_feature <- dplyr::intersect(unique(X6$feature), unique(X6_test$feature))

X6 <- na.omit(X6)
X6 <- subset(X6, feature %in% common_feature)
X6_sp <- sparseMatrix(i = X6$id_num,
                      j = X6$feature,
                      x = X6$value)
dim(X6_sp)

X6_test <- na.omit(X6_test)
X6_test <- subset(X6_test, feature %in% common_feature)
X6_test_sp <- sparseMatrix(i = X6_test$id_num,
                           j = X6_test$feature,
                           x = X6_test$value)
dim(X6_test_sp)

dX6 <- xgb.DMatrix(X6_sp, label = y6, missing = -99999)
dX6_test <- xgb.DMatrix(X6_test_sp, missing = -99999)

# **************************************
# Predict country_destination
# **************************************
Folds <- 10
X6_cv <- createFolds(1:nrow(X6_sp), k = Folds)
X6_stack <- data.frame()
X6_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X6_id_ <- X6_id[-X6_cv[[i]], ]
  X6_fold_id_ <- X6_id[X6_cv[[i]], ]
  X6_sp_ <- X6_sp[-X6_cv[[i]], ]
  X6_fold_sp_ <- X6_sp[X6_cv[[i]], ]
  y6_ <- y6[-X6_cv[[i]]]
  # y6_fold_ <- y6[X6_cv[[i]]]
  
  dX6_ <- xgb.DMatrix(X6_sp_, label = y6_)
  dX6_fold_ <- xgb.DMatrix(X6_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y6),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 1.0,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX6_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X6_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX6_, 
                 param=param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y6_fold_ <- predict(xgb, dX6_fold_)
  y6_fold_mat <- matrix(y6_fold_, nrow=nrow(X6_fold_sp_), ncol=n_distinct(y6_), byrow=T)
  y6_fold_df <- as.data.frame(y6_fold_mat)
  country_destination_label <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
  names(y6_fold_df) <- paste0("pred3_", country_destination_label)
  y6_fold_df <- mutate(y6_fold_df,
                       id = unique(X6_fold_id_$id))
  y6_fold_df <- y6_fold_df[c("id", paste0("pred3_", country_destination_label))]
  X6_stack <- bind_rows(X6_stack,
                        y6_fold_df)
  
  y6_test_ <- predict(xgb, dX6_test)
  y6_test_mat <- matrix(y6_test_, nrow=nrow(X6_test_sp), ncol=n_distinct(y6_), byrow=T)
  y6_test_df <- as.data.frame(y6_test_mat)
  names(y6_test_df) <- paste0("pred3_", country_destination_label)
  y6_test_df <- mutate(y6_test_df,
                       id = unique(X6_test_id$id))
  y6_test_df <- y6_test_df[c("id", paste0("pred3_", country_destination_label))]
  X6_test_stack <- bind_rows(X6_test_stack,
                             y6_test_df)
}
X6_test_stack <- X6_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X6_stack <- melt.data.table(as.data.table(X6_stack))
X6_stack <- data.frame(X6_stack)
names(X6_stack) <- c("id", "feature", "value")

X6_test_stack <- melt.data.table(as.data.table(X6_test_stack))
X6_test_stack <- data.frame(X6_test_stack)
names(X6_test_stack) <- c("id", "feature", "value")

saveRDS(X6_stack, paste0("cache/",folder,"/X6_stack.RData"))
saveRDS(X6_test_stack, paste0("cache/",folder,"/X6_test_stack.RData"))
gc()

# **************************************
# make dataset for predict country_destination
# **************************************
X7_all <- subset(df_all_feats, is.na(value)==F)

X7_all$feature_name <- X7_all$feature
X7_all$feature <- as.numeric(as.factor(X7_all$feature))
X7 <- subset(X7_all, id %in% subset(df_all, dac_yearmonth %in% c("201307", "201308", "201309"))$id)
X7_test <- subset(X7_all, id %in% subset(df_all, dac_yearmonth %nin% c("201307", "201308", "201309"))$id)
X7_all <- rbind(X7, X7_test)

X7_all_feature <- X7_all[!duplicated(X7_all$feature), c("feature", "feature_name")]
X7_all_feature <- X7_all_feature[order(X7_all_feature$feature),]

X7$id_num <- as.numeric(as.factor(X7$id))
X7_test$id_num <- as.numeric(as.factor(X7_test$id))

X7_id <- X7 %>% distinct(id_num)
X7_id <- data.frame(id = X7_id$id, id_num = X7_id$id_num)
X7_id <- dplyr::left_join(X7_id, labels, by = "id")
X7_id <- X7_id %>%
  dplyr::mutate(
    country_destination_num = recode(country_destination,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")
  ) %>%
  dplyr::arrange(id_num)
y7 <- X7_id$country_destination_num

X7_test_id <- X7_test %>% distinct(id_num)
X7_test_id <- data.frame(id = X7_test_id$id, id_num = X7_test_id$id_num)
X7_test_id <- X7_test_id %>%
  dplyr::arrange(id_num)
y7_test <- rep(NA, nrow(X7_test_id))


common_feature <- dplyr::intersect(unique(X7$feature), unique(X7_test$feature))

X7 <- na.omit(X7)
X7 <- subset(X7, feature %in% common_feature)
X7_sp <- sparseMatrix(i = X7$id_num,
                      j = X7$feature,
                      x = X7$value)
dim(X7_sp)

X7_test <- na.omit(X7_test)
X7_test <- subset(X7_test, feature %in% common_feature)
X7_test_sp <- sparseMatrix(i = X7_test$id_num,
                           j = X7_test$feature,
                           x = X7_test$value)
dim(X7_test_sp)

dX7 <- xgb.DMatrix(X7_sp, label = y7, missing = -99999)
dX7_test <- xgb.DMatrix(X7_test_sp, missing = -99999)

# **************************************
# Predict country_destination
# **************************************
Folds <- 10
X7_cv <- createFolds(1:nrow(X7_sp), k = Folds)
X7_stack <- data.frame()
X7_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X7_id_ <- X7_id[-X7_cv[[i]], ]
  X7_fold_id_ <- X7_id[X7_cv[[i]], ]
  X7_sp_ <- X7_sp[-X7_cv[[i]], ]
  X7_fold_sp_ <- X7_sp[X7_cv[[i]], ]
  y7_ <- y7[-X7_cv[[i]]]
  # y7_fold_ <- y7[X7_cv[[i]]]
  
  dX7_ <- xgb.DMatrix(X7_sp_, label = y7_)
  dX7_fold_ <- xgb.DMatrix(X7_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y7),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 1.0,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX7_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X7_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX7_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y7_fold_ <- predict(xgb, dX7_fold_)
  y7_fold_mat <- matrix(y7_fold_, nrow=nrow(X7_fold_sp_), ncol=n_distinct(y7_), byrow=T)
  y7_fold_df <- as.data.frame(y7_fold_mat)
  country_destination_label <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
  names(y7_fold_df) <- paste0("pred4_", country_destination_label)
  y7_fold_df <- mutate(y7_fold_df,
                       id = unique(X7_fold_id_$id))
  y7_fold_df <- y7_fold_df[c("id", paste0("pred4_", country_destination_label))]
  X7_stack <- bind_rows(X7_stack,
                        y7_fold_df)
  
  y7_test_ <- predict(xgb, dX7_test)
  y7_test_mat <- matrix(y7_test_, nrow=nrow(X7_test_sp), ncol=n_distinct(y7_), byrow=T)
  y7_test_df <- as.data.frame(y7_test_mat)
  names(y7_test_df) <- paste0("pred4_", country_destination_label)
  y7_test_df <- mutate(y7_test_df,
                       id = unique(X7_test_id$id))
  y7_test_df <- y7_test_df[c("id", paste0("pred4_", country_destination_label))]
  X7_test_stack <- bind_rows(X7_test_stack,
                             y7_test_df)
}
X7_test_stack <- X7_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X7_stack <- melt.data.table(as.data.table(X7_stack))
X7_stack <- data.frame(X7_stack)
names(X7_stack) <- c("id", "feature", "value")

X7_test_stack <- melt.data.table(as.data.table(X7_test_stack))
X7_test_stack <- data.frame(X7_test_stack)
names(X7_test_stack) <- c("id", "feature", "value")

saveRDS(X7_stack, paste0("cache/",folder,"/X7_stack.RData"))
saveRDS(X7_test_stack, paste0("cache/",folder,"/X7_test_stack.RData"))
gc()

# **************************************
# make dataset for predict age_cln
# **************************************
X8_all <- subset(df_all_feats, is.na(value)==F)

X8_all <- subset(X8_all, feature %nin% c("age"))
X8_all <- subset(X8_all, feature %nin% c("age_cln"))
X8_all <- subset(X8_all, feature %nin% c("age_cln2"))

X8_all$feature_name <- X8_all$feature
X8_all$feature <- as.numeric(as.factor(X8_all$feature))
X8 <- subset(X8_all, id %in% subset(df_all, is.na(age_cln)==F)$id)
X8_test <- subset(X8_all, id %in% subset(df_all, is.na(age_cln)==T)$id)
X8_all <- rbind(X8, X8_test)

X8_all_feature <- X8_all[!duplicated(X8_all$feature), c("feature", "feature_name")]
X8_all_feature <- X8_all_feature[order(X8_all_feature$feature),]

X8$id_num <- as.numeric(as.factor(X8$id))
X8_test$id_num <- as.numeric(as.factor(X8_test$id))

X8_id <- X8 %>% distinct(id_num)
X8_id <- data.frame(id = X8_id$id, id_num = X8_id$id_num)
X8_id <- dplyr::left_join(X8_id, df_all[c("id", "age_cln")], by = "id")
X8_id <- X8_id %>%
  dplyr::arrange(id_num)
y8 <- X8_id$age_cln

X8_test_id <- X8_test %>% distinct(id_num)
X8_test_id <- data.frame(id = X8_test_id$id, id_num = X8_test_id$id_num)
X8_test_id <- X8_test_id %>%
  dplyr::arrange(id_num)
y8_test <- rep(NA, nrow(X8_test_id))


common_feature <- dplyr::intersect(unique(X8$feature), unique(X8_test$feature))

X8 <- na.omit(X8)
X8 <- subset(X8, feature %in% common_feature)
X8_sp <- sparseMatrix(i = X8$id_num,
                      j = X8$feature,
                      x = X8$value)
dim(X8_sp)

X8_test <- na.omit(X8_test)
X8_test <- subset(X8_test, feature %in% common_feature)
X8_test_sp <- sparseMatrix(i = X8_test$id_num,
                           j = X8_test$feature,
                           x = X8_test$value)
dim(X8_test_sp)

dX8 <- xgb.DMatrix(X8_sp, label = y8, missing = -99999)
dX8_test <- xgb.DMatrix(X8_test_sp, missing = -99999)

# **************************************
# Predict age_cln
# **************************************
Folds <- 10
X8_cv <- createFolds(1:nrow(X8_sp), k = Folds)
X8_stack <- data.frame()
X8_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X8_id_ <- X8_id[-X8_cv[[i]], ]
  X8_fold_id_ <- X8_id[X8_cv[[i]], ]
  X8_sp_ <- X8_sp[-X8_cv[[i]], ]
  X8_fold_sp_ <- X8_sp[X8_cv[[i]], ]
  y8_ <- y8[-X8_cv[[i]]]
  # y8_fold_ <- y8[X8_cv[[i]]]
  
  dX8_ <- xgb.DMatrix(X8_sp_, label = y8_)
  dX8_fold_ <- xgb.DMatrix(X8_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX8_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X8_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX8_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y8_fold_ <- predict(xgb, dX8_fold_)
  y8_fold_df <- data.frame(age_cln_pred = y8_fold_)
  y8_fold_df <- mutate(y8_fold_df,
                       id = unique(X8_fold_id_$id))
  y8_fold_df <- y8_fold_df[c("id", "age_cln_pred")]
  X8_stack <- bind_rows(X8_stack,
                        y8_fold_df)
  
  y8_test_ <- predict(xgb, dX8_test)
  y8_test_df <- data.frame(age_cln_pred = y8_test_)
  y8_test_df <- mutate(y8_test_df,
                       id = unique(X8_test_id$id))
  y8_test_df <- y8_test_df[c("id", "age_cln_pred")]
  X8_test_stack <- bind_rows(X8_test_stack,
                             y8_test_df)
}
X8_test_stack <- X8_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X8_stack <- melt.data.table(as.data.table(X8_stack))
X8_stack <- data.frame(X8_stack)
names(X8_stack) <- c("id", "feature", "value")

X8_test_stack <- melt.data.table(as.data.table(X8_test_stack))
X8_test_stack <- data.frame(X8_test_stack)
names(X8_test_stack) <- c("id", "feature", "value")

saveRDS(X8_stack, paste0("cache/",folder,"/X8_stack.RData"))
saveRDS(X8_test_stack, paste0("cache/",folder,"/X8_test_stack.RData"))
gc()

# **************************************
# make dataset for predict age_cln2
# **************************************
X9_all <- subset(df_all_feats, is.na(value)==F)

X9_all <- subset(X9_all, feature %nin% c("age"))
X9_all <- subset(X9_all, feature %nin% c("age_cln"))
X9_all <- subset(X9_all, feature %nin% c("age_cln2"))

X9_all$feature_name <- X9_all$feature
X9_all$feature <- as.numeric(as.factor(X9_all$feature))
X9 <- subset(X9_all, id %in% subset(df_all, is.na(age_cln2)==F)$id)
X9_test <- subset(X9_all, id %in% subset(df_all, is.na(age_cln2)==T)$id)
X9_all <- rbind(X9, X9_test)

X9_all_feature <- X9_all[!duplicated(X9_all$feature), c("feature", "feature_name")]
X9_all_feature <- X9_all_feature[order(X9_all_feature$feature),]

X9$id_num <- as.numeric(as.factor(X9$id))
X9_test$id_num <- as.numeric(as.factor(X9_test$id))

X9_id <- X9 %>% distinct(id_num)
X9_id <- data.frame(id = X9_id$id, id_num = X9_id$id_num)
X9_id <- dplyr::left_join(X9_id, df_all[c("id", "age_cln2")], by = "id")
X9_id <- X9_id %>%
  dplyr::arrange(id_num)
y9 <- X9_id$age_cln2

X9_test_id <- X9_test %>% distinct(id_num)
X9_test_id <- data.frame(id = X9_test_id$id, id_num = X9_test_id$id_num)
X9_test_id <- X9_test_id %>%
  dplyr::arrange(id_num)
y9_test <- rep(NA, nrow(X9_test_id))


common_feature <- dplyr::intersect(unique(X9$feature), unique(X9_test$feature))

X9 <- na.omit(X9)
X9 <- subset(X9, feature %in% common_feature)
X9_sp <- sparseMatrix(i = X9$id_num,
                      j = X9$feature,
                      x = X9$value)
dim(X9_sp)

X9_test <- na.omit(X9_test)
X9_test <- subset(X9_test, feature %in% common_feature)
X9_test_sp <- sparseMatrix(i = X9_test$id_num,
                           j = X9_test$feature,
                           x = X9_test$value)
dim(X9_test_sp)

dX9 <- xgb.DMatrix(X9_sp, label = y9, missing = -99999)
dX9_test <- xgb.DMatrix(X9_test_sp, missing = -99999)

# **************************************
# Predict age_cln2
# **************************************
Folds <- 10
X9_cv <- createFolds(1:nrow(X9_sp), k = Folds)
X9_stack <- data.frame()
X9_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X9_id_ <- X9_id[-X9_cv[[i]], ]
  X9_fold_id_ <- X9_id[X9_cv[[i]], ]
  X9_sp_ <- X9_sp[-X9_cv[[i]], ]
  X9_fold_sp_ <- X9_sp[X9_cv[[i]], ]
  y9_ <- y9[-X9_cv[[i]]]
  # y9_fold_ <- y9[X9_cv[[i]]]
  
  dX9_ <- xgb.DMatrix(X9_sp_, label = y9_)
  dX9_fold_ <- xgb.DMatrix(X9_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.03,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX9_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X9_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX9_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y9_fold_ <- predict(xgb, dX9_fold_)
  y9_fold_df <- data.frame(age_cln2_pred = y9_fold_)
  y9_fold_df <- mutate(y9_fold_df,
                       id = unique(X9_fold_id_$id))
  y9_fold_df <- y9_fold_df[c("id", "age_cln2_pred")]
  X9_stack <- bind_rows(X9_stack,
                        y9_fold_df)
  
  y9_test_ <- predict(xgb, dX9_test)
  y9_test_df <- data.frame(age_cln2_pred = y9_test_)
  y9_test_df <- mutate(y9_test_df,
                       id = unique(X9_test_id$id))
  y9_test_df <- y9_test_df[c("id", "age_cln2_pred")]
  X9_test_stack <- bind_rows(X9_test_stack,
                             y9_test_df)
}
X9_test_stack <- X9_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X9_stack <- melt.data.table(as.data.table(X9_stack))
X9_stack <- data.frame(X9_stack)
names(X9_stack) <- c("id", "feature", "value")

X9_test_stack <- melt.data.table(as.data.table(X9_test_stack))
X9_test_stack <- data.frame(X9_test_stack)
names(X9_test_stack) <- c("id", "feature", "value")

saveRDS(X9_stack, paste0("cache/",folder,"/X9_stack.RData"))
saveRDS(X9_test_stack, paste0("cache/",folder,"/X9_test_stack.RData"))
gc()

# **************************************
# make dataset for predict distance_km
# **************************************
X10_all <- subset(df_all_feats, is.na(value)==F)

X10_all <- subset(X10_all, feature %nin% df_all_countries_feats$feature)

X10_all$feature_name <- X10_all$feature
X10_all$feature <- as.numeric(as.factor(X10_all$feature))
X10 <- subset(X10_all, id %in% subset(df_all, is.na(distance_km)==F)$id)
X10_test <- subset(X10_all, id %in% subset(df_all, is.na(distance_km)==T)$id)
X10_all <- rbind(X10, X10_test)

X10_all_feature <- X10_all[!duplicated(X10_all$feature), c("feature", "feature_name")]
X10_all_feature <- X10_all_feature[order(X10_all_feature$feature),]

X10$id_num <- as.numeric(as.factor(X10$id))
X10_test$id_num <- as.numeric(as.factor(X10_test$id))

X10_id <- X10 %>% distinct(id_num)
X10_id <- data.frame(id = X10_id$id, id_num = X10_id$id_num)
X10_id <- dplyr::left_join(X10_id, df_all[c("id", "distance_km")], by = "id")
X10_id <- X10_id %>%
  dplyr::arrange(id_num)
y10 <- X10_id$distance_km

X10_test_id <- X10_test %>% distinct(id_num)
X10_test_id <- data.frame(id = X10_test_id$id, id_num = X10_test_id$id_num)
X10_test_id <- X10_test_id %>%
  dplyr::arrange(id_num)
y10_test <- rep(NA, nrow(X10_test_id))


common_feature <- dplyr::intersect(unique(X10$feature), unique(X10_test$feature))

X10 <- na.omit(X10)
X10 <- subset(X10, feature %in% common_feature)
X10_sp <- sparseMatrix(i = X10$id_num,
                       j = X10$feature,
                       x = X10$value)
dim(X10_sp)

X10_test <- na.omit(X10_test)
X10_test <- subset(X10_test, feature %in% common_feature)
X10_test_sp <- sparseMatrix(i = X10_test$id_num,
                            j = X10_test$feature,
                            x = X10_test$value)
dim(X10_test_sp)

dX10 <- xgb.DMatrix(X10_sp, label = y10, missing = -99999)
dX10_test <- xgb.DMatrix(X10_test_sp, missing = -99999)

# **************************************
# Predict distance_km
# **************************************
Folds <- 10
X10_cv <- createFolds(1:nrow(X10_sp), k = Folds)
X10_stack <- data.frame()
X10_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X10_id_ <- X10_id[-X10_cv[[i]], ]
  X10_fold_id_ <- X10_id[X10_cv[[i]], ]
  X10_sp_ <- X10_sp[-X10_cv[[i]], ]
  X10_fold_sp_ <- X10_sp[X10_cv[[i]], ]
  y10_ <- y10[-X10_cv[[i]]]
  # y10_fold_ <- y10[X10_cv[[i]]]
  
  dX10_ <- xgb.DMatrix(X10_sp_, label = y10_)
  dX10_fold_ <- xgb.DMatrix(X10_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX10_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X10_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX10_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y10_fold_ <- predict(xgb, dX10_fold_)
  y10_fold_df <- data.frame(distance_km_pred = y10_fold_)
  y10_fold_df <- mutate(y10_fold_df,
                        id = unique(X10_fold_id_$id))
  y10_fold_df <- y10_fold_df[c("id", "distance_km_pred")]
  X10_stack <- bind_rows(X10_stack,
                         y10_fold_df)
  
  y10_test_ <- predict(xgb, dX10_test)
  y10_test_df <- data.frame(distance_km_pred = y10_test_)
  y10_test_df <- mutate(y10_test_df,
                        id = unique(X10_test_id$id))
  y10_test_df <- y10_test_df[c("id", "distance_km_pred")]
  X10_test_stack <- bind_rows(X10_test_stack,
                              y10_test_df)
}
X10_test_stack <- X10_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X10_stack <- melt.data.table(as.data.table(X10_stack))
X10_stack <- data.frame(X10_stack)
names(X10_stack) <- c("id", "feature", "value")

X10_test_stack <- melt.data.table(as.data.table(X10_test_stack))
X10_test_stack <- data.frame(X10_test_stack)
names(X10_test_stack) <- c("id", "feature", "value")

saveRDS(X10_stack, paste0("cache/",folder,"/X10_stack.RData"))
saveRDS(X10_test_stack, paste0("cache/",folder,"/X10_test_stack.RData"))
gc()

# **************************************
# make dataset for predict destination_km2
# **************************************
X11_all <- subset(df_all_feats, is.na(value)==F)

X11_all <- subset(X11_all, feature %nin% df_all_countries_feats$feature)

X11_all$feature_name <- X11_all$feature
X11_all$feature <- as.numeric(as.factor(X11_all$feature))
X11 <- subset(X11_all, id %in% subset(df_all, is.na(destination_km2)==F)$id)
X11_test <- subset(X11_all, id %in% subset(df_all, is.na(destination_km2)==T)$id)
X11_all <- rbind(X11, X11_test)

X11_all_feature <- X11_all[!duplicated(X11_all$feature), c("feature", "feature_name")]
X11_all_feature <- X11_all_feature[order(X11_all_feature$feature),]

X11$id_num <- as.numeric(as.factor(X11$id))
X11_test$id_num <- as.numeric(as.factor(X11_test$id))

X11_id <- X11 %>% distinct(id_num)
X11_id <- data.frame(id = X11_id$id, id_num = X11_id$id_num)
X11_id <- dplyr::left_join(X11_id, df_all[c("id", "destination_km2")], by = "id")
X11_id <- X11_id %>%
  dplyr::arrange(id_num)
y11 <- X11_id$destination_km2

X11_test_id <- X11_test %>% distinct(id_num)
X11_test_id <- data.frame(id = X11_test_id$id, id_num = X11_test_id$id_num)
X11_test_id <- X11_test_id %>%
  dplyr::arrange(id_num)
y11_test <- rep(NA, nrow(X11_test_id))


common_feature <- dplyr::intersect(unique(X11$feature), unique(X11_test$feature))

X11 <- na.omit(X11)
X11 <- subset(X11, feature %in% common_feature)
X11_sp <- sparseMatrix(i = X11$id_num,
                       j = X11$feature,
                       x = X11$value)
dim(X11_sp)

X11_test <- na.omit(X11_test)
X11_test <- subset(X11_test, feature %in% common_feature)
X11_test_sp <- sparseMatrix(i = X11_test$id_num,
                            j = X11_test$feature,
                            x = X11_test$value)
dim(X11_test_sp)

dX11 <- xgb.DMatrix(X11_sp, label = y11, missing = -99999)
dX11_test <- xgb.DMatrix(X11_test_sp, missing = -99999)

# **************************************
# Predict destination_km2
# **************************************
Folds <- 10
X11_cv <- createFolds(1:nrow(X11_sp), k = Folds)
X11_stack <- data.frame()
X11_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X11_id_ <- X11_id[-X11_cv[[i]], ]
  X11_fold_id_ <- X11_id[X11_cv[[i]], ]
  X11_sp_ <- X11_sp[-X11_cv[[i]], ]
  X11_fold_sp_ <- X11_sp[X11_cv[[i]], ]
  y11_ <- y11[-X11_cv[[i]]]
  # y11_fold_ <- y11[X11_cv[[i]]]
  
  dX11_ <- xgb.DMatrix(X11_sp_, label = y11_)
  dX11_fold_ <- xgb.DMatrix(X11_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX11_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X11_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX11_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y11_fold_ <- predict(xgb, dX11_fold_)
  y11_fold_df <- data.frame(destination_km2_pred = y11_fold_)
  y11_fold_df <- mutate(y11_fold_df,
                        id = unique(X11_fold_id_$id))
  y11_fold_df <- y11_fold_df[c("id", "destination_km2_pred")]
  X11_stack <- bind_rows(X11_stack,
                         y11_fold_df)
  
  y11_test_ <- predict(xgb, dX11_test)
  y11_test_df <- data.frame(destination_km2_pred = y11_test_)
  y11_test_df <- mutate(y11_test_df,
                        id = unique(X11_test_id$id))
  y11_test_df <- y11_test_df[c("id", "destination_km2_pred")]
  X11_test_stack <- bind_rows(X11_test_stack,
                              y11_test_df)
}
X11_test_stack <- X11_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X11_stack <- melt.data.table(as.data.table(X11_stack))
X11_stack <- data.frame(X11_stack)
names(X11_stack) <- c("id", "feature", "value")

X11_test_stack <- melt.data.table(as.data.table(X11_test_stack))
X11_test_stack <- data.frame(X11_test_stack)
names(X11_test_stack) <- c("id", "feature", "value")

saveRDS(X11_stack, paste0("cache/",folder,"/X11_stack.RData"))
saveRDS(X11_test_stack, paste0("cache/",folder,"/X11_test_stack.RData"))
gc()

# **************************************
# make dataset for predict gender
# **************************************
X12_all <- subset(df_all_feats, is.na(value)==F)

X12_all$feature_name <- X12_all$feature
X12_all$feature <- as.numeric(as.factor(X12_all$feature))
X12 <- subset(X12_all, id %in% subset(df_all, gender %in% c("FEMALE", "MALE", "OTHER"))$id)
X12_test <- subset(X12_all, id %in% subset(df_all, gender %in% c("-unknown-"))$id)
X12_all <- rbind(X12, X12_test)

X12_all_feature <- X12_all[!duplicated(X12_all$feature), c("feature", "feature_name")]
X12_all_feature <- X12_all_feature[order(X12_all_feature$feature),]

X12$id_num <- as.numeric(as.factor(X12$id))
X12_test$id_num <- as.numeric(as.factor(X12_test$id))

X12_id <- X12 %>% distinct(id_num)
X12_id <- data.frame(id = X12_id$id, id_num = X12_id$id_num)
X12_id <- dplyr::left_join(X12_id, df_all[c("id", "gender")], by = "id")
X12_id <- X12_id %>%
  dplyr::mutate(
    gender_num = recode(gender,"'FEMALE'=0; 'MALE'=1; 'OTHER'=2")
  ) %>%
  dplyr::arrange(id_num)
y12 <- X12_id$gender_num

X12_test_id <- X12_test %>% distinct(id_num)
X12_test_id <- data.frame(id = X12_test_id$id, id_num = X12_test_id$id_num)
X12_test_id <- X12_test_id %>%
  dplyr::arrange(id_num)
y12_test <- rep(NA, nrow(X12_test_id))


common_feature <- dplyr::intersect(unique(X12$feature), unique(X12_test$feature))

X12 <- na.omit(X12)
X12 <- subset(X12, feature %in% common_feature)
X12_sp <- sparseMatrix(i = X12$id_num,
                       j = X12$feature,
                       x = X12$value)
dim(X12_sp)

X12_test <- na.omit(X12_test)
X12_test <- subset(X12_test, feature %in% common_feature)
X12_test_sp <- sparseMatrix(i = X12_test$id_num,
                            j = X12_test$feature,
                            x = X12_test$value)
dim(X12_test_sp)

dX12 <- xgb.DMatrix(X12_sp, label = y12, missing = -99999)
dX12_test <- xgb.DMatrix(X12_test_sp, missing = -99999)

# **************************************
# Predict gender
# **************************************

Folds <- 10
X12_cv <- createFolds(1:nrow(X12_sp), k = Folds)
X12_stack <- data.frame()
X12_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X12_id_ <- X12_id[-X12_cv[[i]], ]
  X12_fold_id_ <- X12_id[X12_cv[[i]], ]
  X12_sp_ <- X12_sp[-X12_cv[[i]], ]
  X12_fold_sp_ <- X12_sp[X12_cv[[i]], ]
  y12_ <- y12[-X12_cv[[i]]]
  # y12_fold_ <- y12[X12_cv[[i]]]
  
  dX12_ <- xgb.DMatrix(X12_sp_, label = y12_)
  dX12_fold_ <- xgb.DMatrix(X12_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y12),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX12_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X12_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX12_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y12_fold_ <- predict(xgb, dX12_fold_)
  y12_fold_mat <- matrix(y12_fold_, nrow=nrow(X12_fold_sp_), ncol=n_distinct(y12_), byrow=T)
  y12_fold_df <- as.data.frame(y12_fold_mat)
  gender_label <- c('FEMALE','MALE','OTHER')
  names(y12_fold_df) <- paste0("gender_pred_", gender_label)
  y12_fold_df <- mutate(y12_fold_df,
                        id = unique(X12_fold_id_$id))
  y12_fold_df <- y12_fold_df[c("id", paste0("gender_pred_", gender_label))]
  X12_stack <- bind_rows(X12_stack,
                         y12_fold_df)
  
  y12_test_ <- predict(xgb, dX12_test)
  y12_test_mat <- matrix(y12_test_, nrow=nrow(X12_test_sp), ncol=n_distinct(y12_), byrow=T)
  y12_test_df <- as.data.frame(y12_test_mat)
  names(y12_test_df) <- paste0("gender_pred_", gender_label)
  y12_test_df <- mutate(y12_test_df,
                        id = unique(X12_test_id$id))
  y12_test_df <- y12_test_df[c("id", paste0("gender_pred_", gender_label))]
  X12_test_stack <- bind_rows(X12_test_stack,
                              y12_test_df)
}
X12_test_stack <- X12_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X12_stack <- melt.data.table(as.data.table(X12_stack))
X12_stack <- data.frame(X12_stack)
names(X12_stack) <- c("id", "feature", "value")

X12_test_stack <- melt.data.table(as.data.table(X12_test_stack))
X12_test_stack <- data.frame(X12_test_stack)
names(X12_test_stack) <- c("id", "feature", "value")

saveRDS(X12_stack, paste0("cache/",folder,"/X12_stack.RData"))
saveRDS(X12_test_stack, paste0("cache/",folder,"/X12_test_stack.RData"))
gc()

# **************************************
# make dataset for predict dfb_tfa_lag_flg
# **************************************
X13_all <- subset(df_all_feats, is.na(value)==F)

X13_all$feature_name <- X13_all$feature
X13_all$feature <- as.numeric(as.factor(X13_all$feature))
X13 <- subset(X13_all, id %in% subset(df_all, dac_yearmonth %nin% c("201407", "201408", "201409"))$id)
X13_test <- subset(X13_all, id %in% subset(df_all, dac_yearmonth %in% c("201407", "201408", "201409"))$id)
X13_all <- rbind(X13, X13_test)

X13_all_feature <- X13_all[!duplicated(X13_all$feature), c("feature", "feature_name")]
X13_all_feature <- X13_all_feature[order(X13_all_feature$feature),]

X13$id_num <- as.numeric(as.factor(X13$id))
X13_test$id_num <- as.numeric(as.factor(X13_test$id))

X13_id <- X13 %>% distinct(id_num)
X13_id <- data.frame(id = X13_id$id, id_num = X13_id$id_num)
X13_id <- dplyr::left_join(X13_id, df_all[c("id", "dfb_tfa_lag_flg")], by = "id")
X13_id <- X13_id %>%
  dplyr::arrange(id_num)
y13 <- X13_id$dfb_tfa_lag_flg

X13_test_id <- X13_test %>% distinct(id_num)
X13_test_id <- data.frame(id = X13_test_id$id, id_num = X13_test_id$id_num)
X13_test_id <- X13_test_id %>%
  dplyr::arrange(id_num)
y13_test <- rep(NA, nrow(X13_test_id))


common_feature <- dplyr::intersect(unique(X13$feature), unique(X13_test$feature))

X13 <- na.omit(X13)
X13 <- subset(X13, feature %in% common_feature)
X13_sp <- sparseMatrix(i = X13$id_num,
                       j = X13$feature,
                       x = X13$value)
dim(X13_sp)

X13_test <- na.omit(X13_test)
X13_test <- subset(X13_test, feature %in% common_feature)
X13_test_sp <- sparseMatrix(i = X13_test$id_num,
                            j = X13_test$feature,
                            x = X13_test$value)
dim(X13_test_sp)

dX13 <- xgb.DMatrix(X13_sp, label = y13, missing = -99999)
dX13_test <- xgb.DMatrix(X13_test_sp, missing = -99999)

# **************************************
# Predict dfb_tfa_lag_flg
# **************************************

Folds <- 10
X13_cv <- createFolds(1:nrow(X13_sp), k = Folds)
X13_stack <- data.frame()
X13_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X13_id_ <- X13_id[-X13_cv[[i]], ]
  X13_fold_id_ <- X13_id[X13_cv[[i]], ]
  X13_sp_ <- X13_sp[-X13_cv[[i]], ]
  X13_fold_sp_ <- X13_sp[X13_cv[[i]], ]
  y13_ <- y13[-X13_cv[[i]]]
  # y13_fold_ <- y13[X13_cv[[i]]]
  
  dX13_ <- xgb.DMatrix(X13_sp_, label = y13_)
  dX13_fold_ <- xgb.DMatrix(X13_fold_sp_)
  
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = n_distinct(y13_),
                "eta" = 0.01,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param = param,
                    data = dX13_, 
                    nfold = Folds,
                    nrounds = cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X13_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX13_, 
                 param = param,
                 nround = which.min(bst.cv$test.mlogloss.mean)
  )
  
  # predict values in test set
  y13_fold_ <- predict(xgb, dX13_fold_)
  y13_fold_mat <- matrix(y13_fold_, nrow=nrow(X13_fold_sp_), ncol=n_distinct(y13_), byrow=T)
  y13_fold_df <- as.data.frame(y13_fold_mat)
  names(y13_fold_df) <- paste0("dfb_tfa_lag_flg_", 1:n_distinct(y13_))
  y13_fold_df <- mutate(y13_fold_df,
                        id = unique(X13_fold_id_$id))
  y13_fold_df <- y13_fold_df[c("id", paste0("dfb_tfa_lag_flg_", 1:n_distinct(y13_)))]
  X13_stack <- bind_rows(X13_stack,
                         y13_fold_df)
  
  y13_test_ <- predict(xgb, dX13_test)
  y13_test_mat <- matrix(y13_test_, nrow=nrow(X13_test_sp), ncol=n_distinct(y13_), byrow=T)
  y13_test_df <- as.data.frame(y13_test_mat)
  names(y13_test_df) <- paste0("dfb_tfa_lag_flg_", 1:n_distinct(y13_))
  y13_test_df <- mutate(y13_test_df,
                        id = unique(X13_test_id$id))
  y13_test_df <- y13_test_df[c("id", paste0("dfb_tfa_lag_flg_", 1:n_distinct(y13_)))]
  X13_test_stack <- bind_rows(X13_test_stack,
                              y13_test_df)
}
X13_test_stack <- X13_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X13_stack <- melt.data.table(as.data.table(X13_stack))
X13_stack <- data.frame(X13_stack)
names(X13_stack) <- c("id", "feature", "value")

X13_test_stack <- melt.data.table(as.data.table(X13_test_stack))
X13_test_stack <- data.frame(X13_test_stack)
names(X13_test_stack) <- c("id", "feature", "value")

saveRDS(X13_stack, paste0("cache/",folder,"/X13_stack.RData"))
saveRDS(X13_test_stack, paste0("cache/",folder,"/X13_test_stack.RData"))
gc()

# **************************************
# make dataset for predict dfb_dac_lag
# **************************************
X14_all <- subset(df_all_feats, is.na(value)==F)

X14_all$feature_name <- X14_all$feature
X14_all$feature <- as.numeric(as.factor(X14_all$feature))
X14 <- subset(X14_all, id %in% subset(df_all, is.na(dfb_dac_lag)==F)$id)
X14_test <- subset(X14_all, id %in% subset(df_all, is.na(dfb_dac_lag)==T)$id)
X14_all <- rbind(X14, X14_test)

X14_all_feature <- X14_all[!duplicated(X14_all$feature), c("feature", "feature_name")]
X14_all_feature <- X14_all_feature[order(X14_all_feature$feature),]

X14$id_num <- as.numeric(as.factor(X14$id))
X14_test$id_num <- as.numeric(as.factor(X14_test$id))

X14_id <- X14 %>% distinct(id_num)
X14_id <- data.frame(id = X14_id$id, id_num = X14_id$id_num)
X14_id <- dplyr::left_join(X14_id, df_all[c("id", "dfb_dac_lag")], by = "id")
X14_id <- X14_id %>%
  dplyr::arrange(id_num)
y14 <- X14_id$dfb_dac_lag

X14_test_id <- X14_test %>% distinct(id_num)
X14_test_id <- data.frame(id = X14_test_id$id, id_num = X14_test_id$id_num)
X14_test_id <- X14_test_id %>%
  dplyr::arrange(id_num)
y14_test <- rep(NA, nrow(X14_test_id))


common_feature <- dplyr::intersect(unique(X14$feature), unique(X14_test$feature))

X14 <- na.omit(X14)
X14 <- subset(X14, feature %in% common_feature)
X14_sp <- sparseMatrix(i = X14$id_num,
                       j = X14$feature,
                       x = X14$value)
dim(X14_sp)

X14_test <- na.omit(X14_test)
X14_test <- subset(X14_test, feature %in% common_feature)
X14_test_sp <- sparseMatrix(i = X14_test$id_num,
                            j = X14_test$feature,
                            x = X14_test$value)
dim(X14_test_sp)

dX14 <- xgb.DMatrix(X14_sp, label = y14, missing = -99999)
dX14_test <- xgb.DMatrix(X14_test_sp, missing = -99999)

# **************************************
# Predict dfb_dac_lag
# **************************************
Folds <- 10
X14_cv <- createFolds(1:nrow(X14_sp), k = Folds)
X14_stack <- data.frame()
X14_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X14_id_ <- X14_id[-X14_cv[[i]], ]
  X14_fold_id_ <- X14_id[X14_cv[[i]], ]
  X14_sp_ <- X14_sp[-X14_cv[[i]], ]
  X14_fold_sp_ <- X14_sp[X14_cv[[i]], ]
  y14_ <- y14[-X14_cv[[i]]]
  # y14_fold_ <- y14[X14_cv[[i]]]
  
  dX14_ <- xgb.DMatrix(X14_sp_, label = y14_)
  dX14_fold_ <- xgb.DMatrix(X14_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.03,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX14_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X14_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX14_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y14_fold_ <- predict(xgb, dX14_fold_)
  y14_fold_df <- data.frame(dfb_dac_lag_pred = y14_fold_)
  y14_fold_df <- mutate(y14_fold_df,
                        id = unique(X14_fold_id_$id))
  y14_fold_df <- y14_fold_df[c("id", "dfb_dac_lag_pred")]
  X14_stack <- bind_rows(X14_stack,
                         y14_fold_df)
  
  y14_test_ <- predict(xgb, dX14_test)
  y14_test_df <- data.frame(dfb_dac_lag_pred = y14_test_)
  y14_test_df <- mutate(y14_test_df,
                        id = unique(X14_test_id$id))
  y14_test_df <- y14_test_df[c("id", "dfb_dac_lag_pred")]
  X14_test_stack <- bind_rows(X14_test_stack,
                              y14_test_df)
}
X14_test_stack <- X14_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X14_stack <- melt.data.table(as.data.table(X14_stack))
X14_stack <- data.frame(X14_stack)
names(X14_stack) <- c("id", "feature", "value")

X14_test_stack <- melt.data.table(as.data.table(X14_test_stack))
X14_test_stack <- data.frame(X14_test_stack)
names(X14_test_stack) <- c("id", "feature", "value")

saveRDS(X14_stack, paste0("cache/",folder,"/X14_stack.RData"))
saveRDS(X14_test_stack, paste0("cache/",folder,"/X14_test_stack.RData"))
gc()

# **************************************
# make dataset for predict dfb_tfa_lag
# **************************************
X15_all <- subset(df_all_feats, is.na(value)==F)

X15_all$feature_name <- X15_all$feature
X15_all$feature <- as.numeric(as.factor(X15_all$feature))
X15 <- subset(X15_all, id %in% subset(df_all, is.na(dfb_tfa_lag)==F)$id)
X15_test <- subset(X15_all, id %in% subset(df_all, is.na(dfb_tfa_lag)==T)$id)
X15_all <- rbind(X15, X15_test)

X15_all_feature <- X15_all[!duplicated(X15_all$feature), c("feature", "feature_name")]
X15_all_feature <- X15_all_feature[order(X15_all_feature$feature),]

X15$id_num <- as.numeric(as.factor(X15$id))
X15_test$id_num <- as.numeric(as.factor(X15_test$id))

X15_id <- X15 %>% distinct(id_num)
X15_id <- data.frame(id = X15_id$id, id_num = X15_id$id_num)
X15_id <- dplyr::left_join(X15_id, df_all[c("id", "dfb_tfa_lag")], by = "id")
X15_id <- X15_id %>%
  dplyr::arrange(id_num)
y15 <- X15_id$dfb_tfa_lag

X15_test_id <- X15_test %>% distinct(id_num)
X15_test_id <- data.frame(id = X15_test_id$id, id_num = X15_test_id$id_num)
X15_test_id <- X15_test_id %>%
  dplyr::arrange(id_num)
y15_test <- rep(NA, nrow(X15_test_id))


common_feature <- dplyr::intersect(unique(X15$feature), unique(X15_test$feature))

X15 <- na.omit(X15)
X15 <- subset(X15, feature %in% common_feature)
X15_sp <- sparseMatrix(i = X15$id_num,
                       j = X15$feature,
                       x = X15$value)
dim(X15_sp)

X15_test <- na.omit(X15_test)
X15_test <- subset(X15_test, feature %in% common_feature)
X15_test_sp <- sparseMatrix(i = X15_test$id_num,
                            j = X15_test$feature,
                            x = X15_test$value)
dim(X15_test_sp)

dX15 <- xgb.DMatrix(X15_sp, label = y15, missing = -99999)
dX15_test <- xgb.DMatrix(X15_test_sp, missing = -99999)

# **************************************
# Predict dfb_tfa_lag
# **************************************
Folds <- 10
X15_cv <- createFolds(1:nrow(X15_sp), k = Folds)
X15_stack <- data.frame()
X15_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X15_id_ <- X15_id[-X15_cv[[i]], ]
  X15_fold_id_ <- X15_id[X15_cv[[i]], ]
  X15_sp_ <- X15_sp[-X15_cv[[i]], ]
  X15_fold_sp_ <- X15_sp[X15_cv[[i]], ]
  y15_ <- y15[-X15_cv[[i]]]
  # y15_fold_ <- y15[X15_cv[[i]]]
  
  dX15_ <- xgb.DMatrix(X15_sp_, label = y15_)
  dX15_fold_ <- xgb.DMatrix(X15_fold_sp_)
  
  param <- list("objective" = "reg:linear",
                "eval_metric" = "rmse",
                "eta" = 0.03,
                "max_depth" = 6,
                "subsample" = 0.7,
                "colsample_bytree" = 0.3,
                # "lambda" = 1.0,
                "alpha" = 1.0,
                # "min_child_weight" = 6,
                # "gamma" = 10,
                "nthread" = 24)
  
  if (i == 1){
    # Run Cross Valication
    cv.nround = 3000
    bst.cv = xgb.cv(param=param,
                    data = dX15_, 
                    nfold = Folds,
                    nrounds=cv.nround,
                    early.stop.round = 10)
    saveRDS(bst.cv, paste0("cache/",folder,"/X15_bst_cv.RData"))
  }
  
  # train xgboost
  xgb <- xgboost(data = dX15_, 
                 param = param,
                 nround = which.min(bst.cv$test.rmse.mean)
  )
  
  # predict values in test set
  y15_fold_ <- predict(xgb, dX15_fold_)
  y15_fold_df <- data.frame(dfb_tfa_lag_pred = y15_fold_)
  y15_fold_df <- mutate(y15_fold_df,
                        id = unique(X15_fold_id_$id))
  y15_fold_df <- y15_fold_df[c("id", "dfb_tfa_lag_pred")]
  X15_stack <- bind_rows(X15_stack,
                         y15_fold_df)
  
  y15_test_ <- predict(xgb, dX15_test)
  y15_test_df <- data.frame(dfb_tfa_lag_pred = y15_test_)
  y15_test_df <- mutate(y15_test_df,
                        id = unique(X15_test_id$id))
  y15_test_df <- y15_test_df[c("id", "dfb_tfa_lag_pred")]
  X15_test_stack <- bind_rows(X15_test_stack,
                              y15_test_df)
}
X15_test_stack <- X15_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X15_stack <- melt.data.table(as.data.table(X15_stack))
X15_stack <- data.frame(X15_stack)
names(X15_stack) <- c("id", "feature", "value")

X15_test_stack <- melt.data.table(as.data.table(X15_test_stack))
X15_test_stack <- data.frame(X15_test_stack)
names(X15_test_stack) <- c("id", "feature", "value")

saveRDS(X15_stack, paste0("cache/",folder,"/X15_stack.RData"))
saveRDS(X15_test_stack, paste0("cache/",folder,"/X15_test_stack.RData"))
gc()

# **************************************
# make dataset for predict age_cln
# **************************************
X16_all <- subset(df_all_feats, is.na(value)==F)

X16_all <- subset(X16_all, feature %nin% c("age"))
X16_all <- subset(X16_all, feature %nin% c("age_cln"))
X16_all <- subset(X16_all, feature %nin% c("age_cln2"))

X16_all$feature_name <- X16_all$feature
X16_all$feature <- as.numeric(as.factor(X16_all$feature))
X16 <- subset(X16_all, id %in% subset(df_all, is.na(age_cln)==F)$id)
X16_test <- subset(X16_all, id %in% subset(df_all, is.na(age_cln)==T)$id)
X16_all <- rbind(X16, X16_test)

X16_all_feature <- X16_all[!duplicated(X16_all$feature), c("feature", "feature_name")]
X16_all_feature <- X16_all_feature[order(X16_all_feature$feature),]

X16$id_num <- as.numeric(as.factor(X16$id))
X16_test$id_num <- as.numeric(as.factor(X16_test$id))

X16_id <- X16 %>% distinct(id_num)
X16_id <- data.frame(id = X16_id$id, id_num = X16_id$id_num)
X16_id <- dplyr::left_join(X16_id, df_all[c("id", "age_cln")], by = "id")
X16_id <- X16_id %>%
  dplyr::arrange(id_num)
y16 <- X16_id$age_cln

X16_test_id <- X16_test %>% distinct(id_num)
X16_test_id <- data.frame(id = X16_test_id$id, id_num = X16_test_id$id_num)
X16_test_id <- X16_test_id %>%
  dplyr::arrange(id_num)
y16_test <- rep(NA, nrow(X16_test_id))


common_feature <- dplyr::intersect(unique(X16$feature), unique(X16_test$feature))

X16 <- na.omit(X16)
X16 <- subset(X16, feature %in% common_feature)
X16_sp <- sparseMatrix(i = X16$id_num,
                       j = X16$feature,
                       x = X16$value)
dim(X16_sp)

X16_test <- na.omit(X16_test)
X16_test <- subset(X16_test, feature %in% common_feature)
X16_test_sp <- sparseMatrix(i = X16_test$id_num,
                            j = X16_test$feature,
                            x = X16_test$value)
dim(X16_test_sp)

dX16 <- xgb.DMatrix(X16_sp, label = y16, missing = -99999)
dX16_test <- xgb.DMatrix(X16_test_sp, missing = -99999)

# **************************************
# Predict age_cln
# **************************************
Folds <- 10
X16_cv <- createFolds(1:nrow(X16_sp), k = Folds)
X16_stack <- data.frame()
X16_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X16_id_ <- X16_id[-X16_cv[[i]], ]
  X16_fold_id_ <- X16_id[X16_cv[[i]], ]
  X16_sp_ <- X16_sp[-X16_cv[[i]], ]
  X16_fold_sp_ <- X16_sp[X16_cv[[i]], ]
  y16_ <- y16[-X16_cv[[i]]]
  # y16_fold_ <- y16[X16_cv[[i]]]
  
  cvfit = cv.glmnet(X16_sp_, y16_, type.measure = "mse", nfolds = 5)
  saveRDS(cvfit, paste0("cache/",folder,"/X16_",i,"_cvfit.RData"))
  print(min(sqrt(cvfit$cvm)))
  
  y16_fold_ <- predict(cvfit, newx = X16_fold_sp_, s = "lambda.min")
  
  y16_fold_df <- data.frame(glm_age_cln_pred = y16_fold_[, 1])
  y16_fold_df <- mutate(y16_fold_df,
                        id = unique(X16_fold_id_$id))
  y16_fold_df <- y16_fold_df[c("id", "glm_age_cln_pred")]
  X16_stack <- bind_rows(X16_stack,
                         y16_fold_df)
  
  y16_test_ <- predict(cvfit, newx = X16_test_sp, s = "lambda.min")
  y16_test_df <- data.frame(glm_age_cln_pred = y16_test_[, 1])
  y16_test_df <- mutate(y16_test_df,
                        id = unique(X16_test_id$id))
  y16_test_df <- y16_test_df[c("id", "glm_age_cln_pred")]
  X16_test_stack <- bind_rows(X16_test_stack,
                              y16_test_df)
}
X16_test_stack <- X16_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X16_stack <- melt.data.table(as.data.table(X16_stack))
X16_stack <- data.frame(X16_stack)
names(X16_stack) <- c("id", "feature", "value")

X16_test_stack <- melt.data.table(as.data.table(X16_test_stack))
X16_test_stack <- data.frame(X16_test_stack)
names(X16_test_stack) <- c("id", "feature", "value")

saveRDS(X16_stack, paste0("cache/",folder,"/X16_stack.RData"))
saveRDS(X16_test_stack, paste0("cache/",folder,"/X16_test_stack.RData"))
gc()

# **************************************
# make dataset for predict age_cln2
# **************************************
X17_all <- subset(df_all_feats, is.na(value)==F)

X17_all <- subset(X17_all, feature %nin% c("age"))
X17_all <- subset(X17_all, feature %nin% c("age_cln"))
X17_all <- subset(X17_all, feature %nin% c("age_cln2"))

X17_all$feature_name <- X17_all$feature
X17_all$feature <- as.numeric(as.factor(X17_all$feature))
X17 <- subset(X17_all, id %in% subset(df_all, is.na(age_cln2)==F)$id)
X17_test <- subset(X17_all, id %in% subset(df_all, is.na(age_cln2)==T)$id)
X17_all <- rbind(X17, X17_test)

X17_all_feature <- X17_all[!duplicated(X17_all$feature), c("feature", "feature_name")]
X17_all_feature <- X17_all_feature[order(X17_all_feature$feature),]

X17$id_num <- as.numeric(as.factor(X17$id))
X17_test$id_num <- as.numeric(as.factor(X17_test$id))

X17_id <- X17 %>% distinct(id_num)
X17_id <- data.frame(id = X17_id$id, id_num = X17_id$id_num)
X17_id <- dplyr::left_join(X17_id, df_all[c("id", "age_cln2")], by = "id")
X17_id <- X17_id %>%
  dplyr::arrange(id_num)
y17 <- X17_id$age_cln2

X17_test_id <- X17_test %>% distinct(id_num)
X17_test_id <- data.frame(id = X17_test_id$id, id_num = X17_test_id$id_num)
X17_test_id <- X17_test_id %>%
  dplyr::arrange(id_num)
y17_test <- rep(NA, nrow(X17_test_id))


common_feature <- dplyr::intersect(unique(X17$feature), unique(X17_test$feature))

X17 <- na.omit(X17)
X17 <- subset(X17, feature %in% common_feature)
X17_sp <- sparseMatrix(i = X17$id_num,
                       j = X17$feature,
                       x = X17$value)
dim(X17_sp)

X17_test <- na.omit(X17_test)
X17_test <- subset(X17_test, feature %in% common_feature)
X17_test_sp <- sparseMatrix(i = X17_test$id_num,
                            j = X17_test$feature,
                            x = X17_test$value)
dim(X17_test_sp)

dX17 <- xgb.DMatrix(X17_sp, label = y17, missing = -99999)
dX17_test <- xgb.DMatrix(X17_test_sp, missing = -99999)

# **************************************
# Predict age_cln2
# **************************************
Folds <- 10
X17_cv <- createFolds(1:nrow(X17_sp), k = Folds)
X17_stack <- data.frame()
X17_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X17_id_ <- X17_id[-X17_cv[[i]], ]
  X17_fold_id_ <- X17_id[X17_cv[[i]], ]
  X17_sp_ <- X17_sp[-X17_cv[[i]], ]
  X17_fold_sp_ <- X17_sp[X17_cv[[i]], ]
  y17_ <- y17[-X17_cv[[i]]]
  # y17_fold_ <- y17[X17_cv[[i]]]
  
  cvfit = cv.glmnet(X17_sp_, y17_, type.measure = "mse", nfolds = 5)
  saveRDS(cvfit, paste0("cache/",folder,"/X17_",i,"_cvfit.RData"))
  print(min(sqrt(cvfit$cvm)))
  
  y17_fold_ <- predict(cvfit, newx = X17_fold_sp_, s = "lambda.min")
  
  y17_fold_df <- data.frame(glm_age_cln2_pred = y17_fold_[, 1])
  y17_fold_df <- mutate(y17_fold_df,
                        id = unique(X17_fold_id_$id))
  y17_fold_df <- y17_fold_df[c("id", "glm_age_cln2_pred")]
  X17_stack <- bind_rows(X17_stack,
                         y17_fold_df)
  
  y17_test_ <- predict(cvfit, newx = X17_test_sp, s = "lambda.min")
  y17_test_df <- data.frame(glm_age_cln2_pred = y17_test_[, 1])
  y17_test_df <- mutate(y17_test_df,
                        id = unique(X17_test_id$id))
  y17_test_df <- y17_test_df[c("id", "glm_age_cln2_pred")]
  X17_test_stack <- bind_rows(X17_test_stack,
                              y17_test_df)
}
X17_test_stack <- X17_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X17_stack <- melt.data.table(as.data.table(X17_stack))
X17_stack <- data.frame(X17_stack)
names(X17_stack) <- c("id", "feature", "value")

X17_test_stack <- melt.data.table(as.data.table(X17_test_stack))
X17_test_stack <- data.frame(X17_test_stack)
names(X17_test_stack) <- c("id", "feature", "value")

saveRDS(X17_stack, paste0("cache/",folder,"/X17_stack.RData"))
saveRDS(X17_test_stack, paste0("cache/",folder,"/X17_test_stack.RData"))
gc()

# **************************************
# make dataset for predict dfb_dac_lag
# **************************************
X18_all <- subset(df_all_feats, is.na(value)==F)

X18_all$feature_name <- X18_all$feature
X18_all$feature <- as.numeric(as.factor(X18_all$feature))
X18 <- subset(X18_all, id %in% subset(df_all, is.na(dfb_dac_lag)==F)$id)
X18_test <- subset(X18_all, id %in% subset(df_all, is.na(dfb_dac_lag)==T)$id)
X18_all <- rbind(X18, X18_test)

X18_all_feature <- X18_all[!duplicated(X18_all$feature), c("feature", "feature_name")]
X18_all_feature <- X18_all_feature[order(X18_all_feature$feature),]

X18$id_num <- as.numeric(as.factor(X18$id))
X18_test$id_num <- as.numeric(as.factor(X18_test$id))

X18_id <- X18 %>% distinct(id_num)
X18_id <- data.frame(id = X18_id$id, id_num = X18_id$id_num)
X18_id <- dplyr::left_join(X18_id, df_all[c("id", "dfb_dac_lag")], by = "id")
X18_id <- X18_id %>%
  dplyr::arrange(id_num)
y18 <- X18_id$dfb_dac_lag

X18_test_id <- X18_test %>% distinct(id_num)
X18_test_id <- data.frame(id = X18_test_id$id, id_num = X18_test_id$id_num)
X18_test_id <- X18_test_id %>%
  dplyr::arrange(id_num)
y18_test <- rep(NA, nrow(X18_test_id))


common_feature <- dplyr::intersect(unique(X18$feature), unique(X18_test$feature))

X18 <- na.omit(X18)
X18 <- subset(X18, feature %in% common_feature)
X18_sp <- sparseMatrix(i = X18$id_num,
                       j = X18$feature,
                       x = X18$value)
dim(X18_sp)

X18_test <- na.omit(X18_test)
X18_test <- subset(X18_test, feature %in% common_feature)
X18_test_sp <- sparseMatrix(i = X18_test$id_num,
                            j = X18_test$feature,
                            x = X18_test$value)
dim(X18_test_sp)

dX18 <- xgb.DMatrix(X18_sp, label = y18, missing = -99999)
dX18_test <- xgb.DMatrix(X18_test_sp, missing = -99999)

# **************************************
# Predict dfb_dac_lag
# **************************************
Folds <- 10
X18_cv <- createFolds(1:nrow(X18_sp), k = Folds)
X18_stack <- data.frame()
X18_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X18_id_ <- X18_id[-X18_cv[[i]], ]
  X18_fold_id_ <- X18_id[X18_cv[[i]], ]
  X18_sp_ <- X18_sp[-X18_cv[[i]], ]
  X18_fold_sp_ <- X18_sp[X18_cv[[i]], ]
  y18_ <- y18[-X18_cv[[i]]]
  # y18_fold_ <- y18[X18_cv[[i]]]
  
  cvfit = cv.glmnet(X18_sp_, y18_, type.measure = "mse", nfolds = 5)
  saveRDS(cvfit, paste0("cache/",folder,"/X18_",i,"_cvfit.RData"))
  print(min(sqrt(cvfit$cvm)))
  
  y18_fold_ <- predict(cvfit, newx = X18_fold_sp_, s = "lambda.min")
  
  y18_fold_df <- data.frame(glm_dfb_dac_lag_pred = y18_fold_[, 1])
  y18_fold_df <- mutate(y18_fold_df,
                        id = unique(X18_fold_id_$id))
  y18_fold_df <- y18_fold_df[c("id", "glm_dfb_dac_lag_pred")]
  X18_stack <- bind_rows(X18_stack,
                         y18_fold_df)
  
  y18_test_ <- predict(cvfit, newx = X18_test_sp, s = "lambda.min")
  y18_test_df <- data.frame(glm_dfb_dac_lag_pred = y18_test_[, 1])
  y18_test_df <- mutate(y18_test_df,
                        id = unique(X18_test_id$id))
  y18_test_df <- y18_test_df[c("id", "glm_dfb_dac_lag_pred")]
  X18_test_stack <- bind_rows(X18_test_stack,
                              y18_test_df)
}
X18_test_stack <- X18_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X18_stack <- melt.data.table(as.data.table(X18_stack))
X18_stack <- data.frame(X18_stack)
names(X18_stack) <- c("id", "feature", "value")

X18_test_stack <- melt.data.table(as.data.table(X18_test_stack))
X18_test_stack <- data.frame(X18_test_stack)
names(X18_test_stack) <- c("id", "feature", "value")

saveRDS(X18_stack, paste0("cache/",folder,"/X18_stack.RData"))
saveRDS(X18_test_stack, paste0("cache/",folder,"/X18_test_stack.RData"))
gc()


# **************************************
# make dataset for predict dfb_tfa_lag
# **************************************
X19_all <- subset(df_all_feats, is.na(value)==F)

X19_all$feature_name <- X19_all$feature
X19_all$feature <- as.numeric(as.factor(X19_all$feature))
X19 <- subset(X19_all, id %in% subset(df_all, is.na(dfb_tfa_lag)==F)$id)
X19_test <- subset(X19_all, id %in% subset(df_all, is.na(dfb_tfa_lag)==T)$id)
X19_all <- rbind(X19, X19_test)

X19_all_feature <- X19_all[!duplicated(X19_all$feature), c("feature", "feature_name")]
X19_all_feature <- X19_all_feature[order(X19_all_feature$feature),]

X19$id_num <- as.numeric(as.factor(X19$id))
X19_test$id_num <- as.numeric(as.factor(X19_test$id))

X19_id <- X19 %>% distinct(id_num)
X19_id <- data.frame(id = X19_id$id, id_num = X19_id$id_num)
X19_id <- dplyr::left_join(X19_id, df_all[c("id", "dfb_tfa_lag")], by = "id")
X19_id <- X19_id %>%
  dplyr::arrange(id_num)
y19 <- X19_id$dfb_tfa_lag

X19_test_id <- X19_test %>% distinct(id_num)
X19_test_id <- data.frame(id = X19_test_id$id, id_num = X19_test_id$id_num)
X19_test_id <- X19_test_id %>%
  dplyr::arrange(id_num)
y19_test <- rep(NA, nrow(X19_test_id))


common_feature <- dplyr::intersect(unique(X19$feature), unique(X19_test$feature))

X19 <- na.omit(X19)
X19 <- subset(X19, feature %in% common_feature)
X19_sp <- sparseMatrix(i = X19$id_num,
                       j = X19$feature,
                       x = X19$value)
dim(X19_sp)

X19_test <- na.omit(X19_test)
X19_test <- subset(X19_test, feature %in% common_feature)
X19_test_sp <- sparseMatrix(i = X19_test$id_num,
                            j = X19_test$feature,
                            x = X19_test$value)
dim(X19_test_sp)

dX19 <- xgb.DMatrix(X19_sp, label = y19, missing = -99999)
dX19_test <- xgb.DMatrix(X19_test_sp, missing = -99999)

# **************************************
# Predict dfb_tfa_lag
# **************************************
Folds <- 10
X19_cv <- createFolds(1:nrow(X19_sp), k = Folds)
X19_stack <- data.frame()
X19_test_stack <- data.frame()
for(i in 1:Folds){ 
  # i <- 1
  set.seed(123 * i)
  X19_id_ <- X19_id[-X19_cv[[i]], ]
  X19_fold_id_ <- X19_id[X19_cv[[i]], ]
  X19_sp_ <- X19_sp[-X19_cv[[i]], ]
  X19_fold_sp_ <- X19_sp[X19_cv[[i]], ]
  y19_ <- y19[-X19_cv[[i]]]
  # y19_fold_ <- y19[X19_cv[[i]]]
  
  cvfit = cv.glmnet(X19_sp_, y19_, type.measure = "mse", nfolds = 5)
  saveRDS(cvfit, paste0("cache/",folder,"/X19_",i,"_cvfit.RData"))
  print(min(sqrt(cvfit$cvm)))
  
  y19_fold_ <- predict(cvfit, newx = X19_fold_sp_, s = "lambda.min")
  
  y19_fold_df <- data.frame(glm_dfb_tfa_lag_pred = y19_fold_[, 1])
  y19_fold_df <- mutate(y19_fold_df,
                        id = unique(X19_fold_id_$id))
  y19_fold_df <- y19_fold_df[c("id", "glm_dfb_tfa_lag_pred")]
  X19_stack <- bind_rows(X19_stack,
                         y19_fold_df)
  
  y19_test_ <- predict(cvfit, newx = X19_test_sp, s = "lambda.min")
  y19_test_df <- data.frame(glm_dfb_tfa_lag_pred = y19_test_[, 1])
  y19_test_df <- mutate(y19_test_df,
                        id = unique(X19_test_id$id))
  y19_test_df <- y19_test_df[c("id", "glm_dfb_tfa_lag_pred")]
  X19_test_stack <- bind_rows(X19_test_stack,
                              y19_test_df)
}
X19_test_stack <- X19_test_stack %>%
  group_by(id) %>%
  summarise_each(funs(mean))

X19_stack <- melt.data.table(as.data.table(X19_stack))
X19_stack <- data.frame(X19_stack)
names(X19_stack) <- c("id", "feature", "value")

X19_test_stack <- melt.data.table(as.data.table(X19_test_stack))
X19_test_stack <- data.frame(X19_test_stack)
names(X19_test_stack) <- c("id", "feature", "value")

saveRDS(X19_stack, paste0("cache/",folder,"/X19_stack.RData"))
saveRDS(X19_test_stack, paste0("cache/",folder,"/X19_test_stack.RData"))
gc()