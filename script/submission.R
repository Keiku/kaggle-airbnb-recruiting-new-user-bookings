# **************************************
# Submission
# **************************************

X_all_feature <- readRDS(paste0("cache/",folder,"/X_all_feature.RData"))
X_id <- readRDS(paste0("cache/",folder,"/X_id.RData"))
X_test_id <- readRDS(paste0("cache/",folder,"/X_test_id.RData"))
X_train_id <- readRDS(paste0("cache/",folder,"/X_train_id.RData"))
X_valid_id <- readRDS(paste0("cache/",folder,"/X_valid_id.RData"))

sample_submission_NDF <- readRDS(paste0("cache/",folder,"/sample_submission_NDF.RData"))

nDCG_score_df <- readRDS(paste0("cache/",folder,"/test/nDCG_score_df_part2.RData"))
i <- which.max(nDCG_score_df$nDCG)

X_test_predictions <- readRDS(paste0("cache/",folder,"/test/X_test_predictions_part2_",i,".RData"))
predictions <- X_test_predictions[c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')]
predictions <- t(as.matrix(predictions))
rownames(predictions) <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
predictions_top5 <- as.vector(apply(predictions, 2, function(x) names(sort(x)[12:8])))

# create submission 
X_test_submission_list <- list()
for (i in 1:nrow(X_test_id)) {
  X_test_idx <- as.character(X_test_id$id[i])
  X_test_ids <- rep(X_test_idx, 5)
  X_test_submission <- data.frame(id = X_test_ids,
                                  id_num = 1:5)
  X_test_submission_list[[i]] <- X_test_submission
}
X_test_submission <- bind_rows(X_test_submission_list)
X_test_submission$country <- predictions_top5

# create submission 
submission_list <- list()
for (i in 1:nrow(sample_submission_NDF)) {
  submission_idx <- as.character(sample_submission_NDF$id[i])
  submission_ids <- rep(submission_idx, 5)
  submission <- data.frame(id = submission_ids,
                           id_num = 1:5)
  submission_list[[i]] <- submission
}
submission <- bind_rows(submission_list)

submission <- dplyr::left_join(submission,
                               X_test_submission,
                               by = c("id", "id_num"))
submission$id_num <- NULL

# generate submission file
file_name <- "2nd_place_solution"
write.csv(submission, paste0("submit/", folder, "/", file_name,".csv"), quote=FALSE, row.names = FALSE)
system(paste0("7za a submit/", folder, "/",file_name,".csv.7z submit/", folder, "/",file_name,".csv"))