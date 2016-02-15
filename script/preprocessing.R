# **************************************
# load data
# **************************************
df_train = read_csv("data/train_users_2.csv",
                    col_types = cols(
                      timestamp_first_active = col_character()))
df_test = read_csv("data/test_users.csv",
                   col_types = cols(
                     timestamp_first_active = col_character()))
labels = df_train[, c('id', 'country_destination')]
df_test$country_destination = NA

age_gender_bkts <- fread("data/age_gender_bkts.csv", data.table=F)
countries <- fread("data/countries.csv", data.table=F)
sample_submission_NDF <- fread("data/sample_submission_NDF.csv", data.table=F)
sessions <- fread("data/sessions.csv", data.table=F)

# combine train and test data
df_train$dataset <- "train"
df_test$dataset <- "test"
df_all = rbind(df_train, df_test)

# **************************************
# clean age
# **************************************
df_all <- df_all %>%
  dplyr::mutate(
    age_cln = ifelse(age >= 1920, 2015 - age, age),
    age_cln2 = ifelse(age_cln < 14 | age_cln > 100, -1, age_cln),
    age_bucket = cut(age, breaks = c(Min(age_cln), 4, 9, 14, 19, 24,
                                     29, 34, 39, 44, 49, 54,
                                     59, 64, 69, 74, 79, 84,
                                     89, 94, 99, Max(age_cln)
    )),
    age_bucket = mapvalues(age_bucket,
                           from=c("(1,4]", "(4,9]", "(9,14]", "(14,19]",
                                  "(19,24]", "(24,29]", "(29,34]", "(34,39]",
                                  "(39,44]", "(44,49]", "(49,54]", "(54,59]",
                                  "(59,64]", "(64,69]", "(69,74]", "(74,79]",
                                  "(79,84]", "(84,89]", "(89,94]", "(94,99]", "(99,150]"),
                           to=c("0-4", "5-9", "10-14", "15-19",
                                "20-24", "25-29", "30-34", "35-39",
                                "40-44", "45-49", "50-54", "55-59",
                                "60-64", "65-69", "70-74", "75-79",
                                "80-84", "85-89", "90-94", "95-99", "100+"))
  )

# **************************************
# feature using date_first_booking
# **************************************
df_all <- df_all %>%
  separate(date_account_created, into = c("dac_year", "dac_month", "dac_day"), sep = "-", remove=FALSE) %>%
  dplyr::mutate(
    dac_yearmonth = paste0(dac_year, dac_month),
    dac_yearmonthday = as.numeric(paste0(dac_year, dac_month, dac_day)),
    dac_week = as.numeric(format(date_account_created+3, "%U")),
    dac_yearmonthweek = as.numeric(paste0(dac_year, dac_month, formatC(dac_week, width=2, flag="0"))),
    tfa_year = str_sub(timestamp_first_active, 1, 4),
    tfa_month = str_sub(timestamp_first_active, 5, 6),
    tfa_day = str_sub(timestamp_first_active, 7, 8),
    tfa_yearmonth = str_sub(timestamp_first_active, 1, 6),
    tfa_yearmonthday = as.numeric(str_sub(timestamp_first_active, 1, 8)),
    tfa_date = as.Date(paste(tfa_year, tfa_month, tfa_day, sep="-")),
    tfa_week = as.numeric(format(tfa_date+3, "%U")),
    tfa_yearmonthweek = as.numeric(paste0(tfa_year, tfa_month, formatC(tfa_week, width=2, flag="0"))),
    dac_lag = as.numeric(date_account_created - tfa_date),
    dfb_dac_lag = as.numeric(date_first_booking - date_account_created),
    dfb_dac_lag_cut = as.character(cut2(dfb_dac_lag, c(0, 1))),
    dfb_dac_lag_flg = as.numeric(as.factor(ifelse(is.na(dfb_dac_lag_cut)==T, "NA", dfb_dac_lag_cut))) - 1,
    dfb_tfa_lag = as.numeric(date_first_booking - tfa_date),
    dfb_tfa_lag_cut = as.character(cut2(dfb_tfa_lag, c(0, 1))),
    dfb_tfa_lag_flg = as.numeric(as.factor(ifelse(is.na(dfb_tfa_lag_cut)==T, "NA", dfb_tfa_lag_cut))) - 1
  )


# **************************************
# join countries
# **************************************
countries <- dplyr::mutate(countries,
                           language = str_sub(destination_language, 1, 2))
df_all <- df_all %>%
  dplyr::mutate(country_destination = country_destination) %>%
  dplyr::left_join(., countries[c("language", "country_destination", "distance_km", "destination_km2", "language_levenshtein_distance")],
                   by = c("language", "country_destination"))
countries$language <- NULL


# **************************************
# set validation
# **************************************
df_train <- subset(df_all, dataset == "train")
df_train <- df_train %>%
  dplyr::mutate(dataset = ifelse(dac_yearmonth %nin% c("201404", "201405", "201406"), "train", "valid"))
df_test <- subset(df_all, dataset == "test")
df_all = rbind(df_train, df_test)


# **************************************
# stack numeric feature
# **************************************
num_feats <- c(
  "age_cln",
  "age_cln2",
  "dac_year",
  "dac_month",
  "dac_yearmonth",
  "dac_yearmonthday",
  "dac_yearmonthweek",
  "dac_day",
  "dac_week",
  "tfa_year",
  "tfa_month",
  "tfa_yearmonth",
  "tfa_yearmonthday",
  "tfa_yearmonthweek",
  "tfa_day",
  "tfa_week"#,
  # "dac_lag",
  # "dfb_dac_lag",
  # "dfb_tfa_lag"
)
df_all_num_feats <- list()
i <- 1
for(feat in num_feats){
  df_all_num_feats_ <- df_all[c("id", feat)]
  df_all_num_feats_$feature <- feat
  df_all_num_feats_$value <- as.numeric(df_all_num_feats_[[feat]])
  df_all_num_feats_ <- df_all_num_feats_[c("id", "feature", "value")]
  df_all_num_feats[[i]] <- df_all_num_feats_
  i <- i + 1
}
df_all_num_feats <- bind_rows(df_all_num_feats)
print("numeric feature")
print(n_distinct(df_all_num_feats$feature))


# **************************************
# stack categorical feature
# **************************************
ohe_feats = c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser')
df_all_ohe_feats <- list()
i <- 1
n_feats <- 0
for(feat in ohe_feats){
  df_all_ohe_feats_ <- df_all[c("id", feat)]
  df_all_ohe_feats_$feature <- paste(feat, df_all_ohe_feats_[[feat]], sep="_")
  n_feats_ <- n_distinct(df_all_ohe_feats_$feature)
  df_all_ohe_feats_$value <- 1
  df_all_ohe_feats_ <- df_all_ohe_feats_[c("id", "feature", "value")]
  df_all_ohe_feats[[i]] <- df_all_ohe_feats_
  i <- i + 1
  n_feats <- n_feats + n_feats_
}
df_all_ohe_feats <- bind_rows(df_all_ohe_feats)
print("categorical feature")
print(n_feats)


# **************************************
# sessions features
# **************************************
sessions$flg <- 1
sessions <- data.table(sessions)
sessions[, seq := sequence(.N), by = c("user_id")]
sessions[, seq_rev := rev(sequence(.N)), by = c("user_id")]
sessions[, action2 := paste(action, action_type, action_detail, device_type, sep="_"),]

first_execution <- 1

if(first_execution == 1){
  sessions_action_se_sum <- sessions[,list(secs_elapsed_sum = sum(secs_elapsed, na.rm=T)),
                                     by=list(user_id, action)]
  sessions_action_se_sum <- melt.data.table(sessions_action_se_sum)
  sessions_action_se_sum$variable <- NULL
  sessions_action_se_sum <- data.frame(sessions_action_se_sum)
  names(sessions_action_se_sum) <- c("id", "feature", "value")
  sessions_action_se_sum$feature <- paste("action_se_sum", sessions_action_se_sum$feature, sep="_")
  n_distinct(sessions_action_se_sum$feature)
  saveRDS(sessions_action_se_sum, "cache/sessions_action_se_sum.RData")
  
  
  sessions_action_type_se_sum <- sessions[,list(secs_elapsed_sum = sum(secs_elapsed, na.rm=T)),
                                          by=list(user_id, action_type)]
  sessions_action_type_se_sum <- melt.data.table(sessions_action_type_se_sum)
  sessions_action_type_se_sum$variable <- NULL
  sessions_action_type_se_sum <- data.frame(sessions_action_type_se_sum)
  names(sessions_action_type_se_sum) <- c("id", "feature", "value")
  sessions_action_type_se_sum$feature <- paste("action_type_se_sum", sessions_action_type_se_sum$feature, sep="_")
  n_distinct(sessions_action_type_se_sum$feature)
  saveRDS(sessions_action_type_se_sum, "cache/sessions_action_type_se_sum.RData")
  
  
  sessions_action_detail_se_sum <- sessions[,list(secs_elapsed_sum = sum(secs_elapsed, na.rm=T)),
                                            by=list(user_id, action_detail)]
  sessions_action_detail_se_sum <- melt.data.table(sessions_action_detail_se_sum)
  sessions_action_detail_se_sum$variable <- NULL
  sessions_action_detail_se_sum <- data.frame(sessions_action_detail_se_sum)
  names(sessions_action_detail_se_sum) <- c("id", "feature", "value")
  sessions_action_detail_se_sum$feature <- paste("action_detail_se_sum", sessions_action_detail_se_sum$feature, sep="_")
  n_distinct(sessions_action_detail_se_sum$feature)
  saveRDS(sessions_action_detail_se_sum, "cache/sessions_action_detail_se_sum.RData")
  
  
  sessions_device_type_se_sum <- sessions[,list(secs_elapsed_sum = sum(secs_elapsed, na.rm=T)),
                                          by=list(user_id, device_type)]
  sessions_device_type_se_sum <- melt.data.table(sessions_device_type_se_sum)
  sessions_device_type_se_sum$variable <- NULL
  sessions_device_type_se_sum <- data.frame(sessions_device_type_se_sum)
  names(sessions_device_type_se_sum) <- c("id", "feature", "value")
  sessions_device_type_se_sum$feature <- paste("device_type_se_sum", sessions_device_type_se_sum$feature, sep="_")
  n_distinct(sessions_device_type_se_sum$feature)
  saveRDS(sessions_device_type_se_sum, "cache/sessions_device_type_se_sum.RData")
  
  
  sessions_action_flg_sum <- sessions[,list(flg_sum = sum(flg, na.rm=T)),
                                      by=list(user_id, action)]
  sessions_action_flg_sum <- melt.data.table(sessions_action_flg_sum)
  sessions_action_flg_sum$variable <- NULL
  sessions_action_flg_sum <- data.frame(sessions_action_flg_sum)
  names(sessions_action_flg_sum) <- c("id", "feature", "value")
  sessions_action_flg_sum$feature <- paste("action_flg_sum", sessions_action_flg_sum$feature, sep="_")
  n_distinct(sessions_action_flg_sum$feature)
  saveRDS(sessions_action_flg_sum, "cache/sessions_action_flg_sum.RData")
  
  
  sessions_action_type_flg_sum <- sessions[,list(flg_sum = sum(flg, na.rm=T)),
                                           by=list(user_id, action_type)]
  sessions_action_type_flg_sum <- melt.data.table(sessions_action_type_flg_sum)
  sessions_action_type_flg_sum$variable <- NULL
  sessions_action_type_flg_sum <- data.frame(sessions_action_type_flg_sum)
  names(sessions_action_type_flg_sum) <- c("id", "feature", "value")
  sessions_action_type_flg_sum$feature <- paste("action_type_flg_sum", sessions_action_type_flg_sum$feature, sep="_")
  n_distinct(sessions_action_type_flg_sum$feature)
  saveRDS(sessions_action_type_flg_sum, "cache/sessions_action_type_flg_sum.RData")
  
  
  sessions_action_detail_flg_sum <- sessions[,list(flg_sum = sum(flg, na.rm=T)),
                                             by=list(user_id, action_detail)]
  sessions_action_detail_flg_sum <- melt.data.table(sessions_action_detail_flg_sum)
  sessions_action_detail_flg_sum$variable <- NULL
  sessions_action_detail_flg_sum <- data.frame(sessions_action_detail_flg_sum)
  names(sessions_action_detail_flg_sum) <- c("id", "feature", "value")
  sessions_action_detail_flg_sum$feature <- paste("action_detail_flg_sum", sessions_action_detail_flg_sum$feature, sep="_")
  n_distinct(sessions_action_detail_flg_sum$feature)
  saveRDS(sessions_action_detail_flg_sum, "cache/sessions_action_detail_flg_sum.RData")
  
  
  sessions_device_type_flg_sum <- sessions[,list(flg_sum = sum(flg, na.rm=T)),
                                           by=list(user_id, device_type)]
  sessions_device_type_flg_sum <- melt.data.table(sessions_device_type_flg_sum)
  sessions_device_type_flg_sum$variable <- NULL
  sessions_device_type_flg_sum <- data.frame(sessions_device_type_flg_sum)
  names(sessions_device_type_flg_sum) <- c("id", "feature", "value")
  sessions_device_type_flg_sum$feature <- paste("device_type_flg_sum", sessions_device_type_flg_sum$feature, sep="_")
  n_distinct(sessions_device_type_flg_sum$feature)
  saveRDS(sessions_device_type_flg_sum, "cache/sessions_device_type_flg_sum.RData")
  
  
  sessions_action_se_mean <- sessions[,list(secs_elapsed_mean = mean(secs_elapsed, na.rm=T)),
                                      by=list(user_id, action)]
  sessions_action_se_mean <- melt.data.table(sessions_action_se_mean)
  sessions_action_se_mean$variable <- NULL
  sessions_action_se_mean <- data.frame(sessions_action_se_mean)
  names(sessions_action_se_mean) <- c("id", "feature", "value")
  sessions_action_se_mean$feature <- paste("action_se_mean", sessions_action_se_mean$feature, sep="_")
  n_distinct(sessions_action_se_mean$feature)
  saveRDS(sessions_action_se_mean, "cache/sessions_action_se_mean.RData")
  
  
  sessions_action_type_se_mean <- sessions[,list(secs_elapsed_mean = mean(secs_elapsed, na.rm=T)),
                                           by=list(user_id, action_type)]
  sessions_action_type_se_mean <- melt.data.table(sessions_action_type_se_mean)
  sessions_action_type_se_mean$variable <- NULL
  sessions_action_type_se_mean <- data.frame(sessions_action_type_se_mean)
  names(sessions_action_type_se_mean) <- c("id", "feature", "value")
  sessions_action_type_se_mean$feature <- paste("action_type_se_mean", sessions_action_type_se_mean$feature, sep="_")
  n_distinct(sessions_action_type_se_mean$feature)
  saveRDS(sessions_action_type_se_mean, "cache/sessions_action_type_se_mean.RData")
  
  
  sessions_action_detail_se_mean <- sessions[,list(secs_elapsed_mean = mean(secs_elapsed, na.rm=T)),
                                             by=list(user_id, action_detail)]
  sessions_action_detail_se_mean <- melt.data.table(sessions_action_detail_se_mean)
  sessions_action_detail_se_mean$variable <- NULL
  sessions_action_detail_se_mean <- data.frame(sessions_action_detail_se_mean)
  names(sessions_action_detail_se_mean) <- c("id", "feature", "value")
  sessions_action_detail_se_mean$feature <- paste("action_detail_se_mean", sessions_action_detail_se_mean$feature, sep="_")
  n_distinct(sessions_action_detail_se_mean$feature)
  saveRDS(sessions_action_detail_se_mean, "cache/sessions_action_detail_se_mean.RData")
  
  
  sessions_device_type_se_mean <- sessions[,list(secs_elapsed_mean = mean(secs_elapsed, na.rm=T)),
                                           by=list(user_id, device_type)]
  sessions_device_type_se_mean <- melt.data.table(sessions_device_type_se_mean)
  sessions_device_type_se_mean$variable <- NULL
  sessions_device_type_se_mean <- data.frame(sessions_device_type_se_mean)
  names(sessions_device_type_se_mean) <- c("id", "feature", "value")
  sessions_device_type_se_mean$feature <- paste("device_type_se_mean", sessions_device_type_se_mean$feature, sep="_")
  n_distinct(sessions_device_type_se_mean$feature)
  saveRDS(sessions_device_type_se_mean, "cache/sessions_device_type_se_mean.RData")
  
  
  sessions_action_se_sd <- sessions[,list(secs_elapsed_sd = sd(secs_elapsed, na.rm=T)),
                                    by=list(user_id, action)]
  sessions_action_se_sd <- melt.data.table(sessions_action_se_sd)
  sessions_action_se_sd$variable <- NULL
  sessions_action_se_sd <- data.frame(sessions_action_se_sd)
  names(sessions_action_se_sd) <- c("id", "feature", "value")
  sessions_action_se_sd$feature <- paste("action_se_sd", sessions_action_se_sd$feature, sep="_")
  n_distinct(sessions_action_se_sd$feature)
  saveRDS(sessions_action_se_sd, "cache/sessions_action_se_sd.RData")
  
  
  sessions_action_type_se_sd <- sessions[,list(secs_elapsed_sd = sd(secs_elapsed, na.rm=T)),
                                         by=list(user_id, action_type)]
  sessions_action_type_se_sd <- melt.data.table(sessions_action_type_se_sd)
  sessions_action_type_se_sd$variable <- NULL
  sessions_action_type_se_sd <- data.frame(sessions_action_type_se_sd)
  names(sessions_action_type_se_sd) <- c("id", "feature", "value")
  sessions_action_type_se_sd$feature <- paste("action_type_se_sd", sessions_action_type_se_sd$feature, sep="_")
  n_distinct(sessions_action_type_se_sd$feature)
  saveRDS(sessions_action_type_se_sd, "cache/sessions_action_type_se_sd.RData")
  
  
  sessions_action_detail_se_sd <- sessions[,list(secs_elapsed_sd = sd(secs_elapsed, na.rm=T)),
                                           by=list(user_id, action_detail)]
  sessions_action_detail_se_sd <- melt.data.table(sessions_action_detail_se_sd)
  sessions_action_detail_se_sd$variable <- NULL
  sessions_action_detail_se_sd <- data.frame(sessions_action_detail_se_sd)
  names(sessions_action_detail_se_sd) <- c("id", "feature", "value")
  sessions_action_detail_se_sd$feature <- paste("action_detail_se_sd", sessions_action_detail_se_sd$feature, sep="_")
  n_distinct(sessions_action_detail_se_sd$feature)
  saveRDS(sessions_action_detail_se_sd, "cache/sessions_action_detail_se_sd.RData")
  
  
  sessions_device_type_se_sd <- sessions[,list(secs_elapsed_sd = sd(secs_elapsed, na.rm=T)),
                                         by=list(user_id, device_type)]
  sessions_device_type_se_sd <- melt.data.table(sessions_device_type_se_sd)
  sessions_device_type_se_sd$variable <- NULL
  sessions_device_type_se_sd <- data.frame(sessions_device_type_se_sd)
  names(sessions_device_type_se_sd) <- c("id", "feature", "value")
  sessions_device_type_se_sd$feature <- paste("device_type_se_sd", sessions_device_type_se_sd$feature, sep="_")
  n_distinct(sessions_device_type_se_sd$feature)
  saveRDS(sessions_device_type_se_sd, "cache/sessions_device_type_se_sd.RData")
  
  
  sessions_action_se_wrmean <- sessions[,list(secs_elapsed_wrmean = weighted.mean(secs_elapsed, w = 1/seq_rev)),
                                        by=list(user_id, action)]
  sessions_action_se_wrmean <- melt.data.table(sessions_action_se_wrmean)
  sessions_action_se_wrmean$variable <- NULL
  sessions_action_se_wrmean <- data.frame(sessions_action_se_wrmean)
  names(sessions_action_se_wrmean) <- c("id", "feature", "value")
  sessions_action_se_wrmean$feature <- paste("action_se_wrmean", sessions_action_se_wrmean$feature, sep="_")
  n_distinct(sessions_action_se_wrmean$feature)
  saveRDS(sessions_action_se_wrmean, "cache/sessions_action_se_wrmean.RData")
  
  
  sessions_action_type_se_wrmean <- sessions[,list(secs_elapsed_wrmean = weighted.mean(secs_elapsed, w = 1/seq_rev)),
                                             by=list(user_id, action_type)]
  sessions_action_type_se_wrmean <- melt.data.table(sessions_action_type_se_wrmean)
  sessions_action_type_se_wrmean$variable <- NULL
  sessions_action_type_se_wrmean <- data.frame(sessions_action_type_se_wrmean)
  names(sessions_action_type_se_wrmean) <- c("id", "feature", "value")
  sessions_action_type_se_wrmean$feature <- paste("action_type_se_wrmean", sessions_action_type_se_wrmean$feature, sep="_")
  n_distinct(sessions_action_type_se_wrmean$feature)
  saveRDS(sessions_action_type_se_wrmean, "cache/sessions_action_type_se_wrmean.RData")
  
  
  sessions_action_detail_se_wrmean <- sessions[,list(secs_elapsed_wrmean = weighted.mean(secs_elapsed, w = 1/seq_rev)),
                                               by=list(user_id, action_detail)]
  sessions_action_detail_se_wrmean <- melt.data.table(sessions_action_detail_se_wrmean)
  sessions_action_detail_se_wrmean$variable <- NULL
  sessions_action_detail_se_wrmean <- data.frame(sessions_action_detail_se_wrmean)
  names(sessions_action_detail_se_wrmean) <- c("id", "feature", "value")
  sessions_action_detail_se_wrmean$feature <- paste("action_detail_se_wrmean", sessions_action_detail_se_wrmean$feature, sep="_")
  n_distinct(sessions_action_detail_se_wrmean$feature)
  saveRDS(sessions_action_detail_se_wrmean, "cache/sessions_action_detail_se_wrmean.RData")
  
  
  sessions_device_type_se_wrmean <- sessions[,list(secs_elapsed_wrmean = weighted.mean(secs_elapsed, w = 1/seq_rev)),
                                             by=list(user_id, device_type)]
  sessions_device_type_se_wrmean <- melt.data.table(sessions_device_type_se_wrmean)
  sessions_device_type_se_wrmean$variable <- NULL
  sessions_device_type_se_wrmean <- data.frame(sessions_device_type_se_wrmean)
  names(sessions_device_type_se_wrmean) <- c("id", "feature", "value")
  sessions_device_type_se_wrmean$feature <- paste("device_type_se_wrmean", sessions_device_type_se_wrmean$feature, sep="_")
  n_distinct(sessions_device_type_se_wrmean$feature)
  saveRDS(sessions_device_type_se_wrmean, "cache/sessions_device_type_se_wrmean.RData")
  
  
  sessions_action_se_wmean <- sessions[,list(secs_elapsed_wmean = weighted.mean(secs_elapsed, w = 1/seq)),
                                       by=list(user_id, action)]
  sessions_action_se_wmean <- melt.data.table(sessions_action_se_wmean)
  sessions_action_se_wmean$variable <- NULL
  sessions_action_se_wmean <- data.frame(sessions_action_se_wmean)
  names(sessions_action_se_wmean) <- c("id", "feature", "value")
  sessions_action_se_wmean$feature <- paste("action_se_wmean", sessions_action_se_wmean$feature, sep="_")
  n_distinct(sessions_action_se_wmean$feature)
  saveRDS(sessions_action_se_wmean, "cache/sessions_action_se_wmean.RData")
  
  
  sessions_action_type_se_wmean <- sessions[,list(secs_elapsed_wmean = weighted.mean(secs_elapsed, w = 1/seq)),
                                            by=list(user_id, action_type)]
  sessions_action_type_se_wmean <- melt.data.table(sessions_action_type_se_wmean)
  sessions_action_type_se_wmean$variable <- NULL
  sessions_action_type_se_wmean <- data.frame(sessions_action_type_se_wmean)
  names(sessions_action_type_se_wmean) <- c("id", "feature", "value")
  sessions_action_type_se_wmean$feature <- paste("action_type_se_wmean", sessions_action_type_se_wmean$feature, sep="_")
  n_distinct(sessions_action_type_se_wmean$feature)
  saveRDS(sessions_action_type_se_wmean, "cache/sessions_action_type_se_wmean.RData")
  
  
  sessions_action_detail_se_wmean <- sessions[,list(secs_elapsed_wmean = weighted.mean(secs_elapsed, w = 1/seq)),
                                              by=list(user_id, action_detail)]
  sessions_action_detail_se_wmean <- melt.data.table(sessions_action_detail_se_wmean)
  sessions_action_detail_se_wmean$variable <- NULL
  sessions_action_detail_se_wmean <- data.frame(sessions_action_detail_se_wmean)
  names(sessions_action_detail_se_wmean) <- c("id", "feature", "value")
  sessions_action_detail_se_wmean$feature <- paste("action_detail_se_wmean", sessions_action_detail_se_wmean$feature, sep="_")
  n_distinct(sessions_action_detail_se_wmean$feature)
  saveRDS(sessions_action_detail_se_wmean, "cache/sessions_action_detail_se_wmean.RData")
  
  
  sessions_device_type_se_wmean <- sessions[,list(secs_elapsed_wmean = weighted.mean(secs_elapsed, w = 1/seq)),
                                            by=list(user_id, device_type)]
  sessions_device_type_se_wmean <- melt.data.table(sessions_device_type_se_wmean)
  sessions_device_type_se_wmean$variable <- NULL
  sessions_device_type_se_wmean <- data.frame(sessions_device_type_se_wmean)
  names(sessions_device_type_se_wmean) <- c("id", "feature", "value")
  sessions_device_type_se_wmean$feature <- paste("device_type_se_wmean", sessions_device_type_se_wmean$feature, sep="_")
  n_distinct(sessions_device_type_se_wmean$feature)
  saveRDS(sessions_device_type_se_wmean, "cache/sessions_device_type_se_wmean.RData")
}

sessions_action_se_sum <- readRDS("cache/sessions_action_se_sum.RData")
sessions_action_type_se_sum <- readRDS("cache/sessions_action_type_se_sum.RData")
sessions_action_detail_se_sum <- readRDS("cache/sessions_action_detail_se_sum.RData")
sessions_device_type_se_sum <- readRDS("cache/sessions_device_type_se_sum.RData")
sessions_action_flg_sum <- readRDS("cache/sessions_action_flg_sum.RData")
sessions_action_type_flg_sum <- readRDS("cache/sessions_action_type_flg_sum.RData")
sessions_action_detail_flg_sum <- readRDS("cache/sessions_action_detail_flg_sum.RData")
sessions_device_type_flg_sum <- readRDS("cache/sessions_device_type_flg_sum.RData")
sessions_action_se_mean <- readRDS("cache/sessions_action_se_mean.RData")
sessions_action_type_se_mean <- readRDS("cache/sessions_action_type_se_mean.RData")
sessions_action_detail_se_mean <- readRDS("cache/sessions_action_detail_se_mean.RData")
sessions_device_type_se_mean <- readRDS("cache/sessions_device_type_se_mean.RData")
sessions_action_se_sd <- readRDS("cache/sessions_action_se_sd.RData")
sessions_action_type_se_sd <- readRDS("cache/sessions_action_type_se_sd.RData")
sessions_action_detail_se_sd <- readRDS("cache/sessions_action_detail_se_sd.RData")
sessions_device_type_se_sd <- readRDS("cache/sessions_device_type_se_sd.RData")
sessions_action_se_wrmean <- readRDS("cache/sessions_action_se_wrmean.RData")
sessions_action_type_se_wrmean <- readRDS("cache/sessions_action_type_se_wrmean.RData")
sessions_action_detail_se_wrmean <- readRDS("cache/sessions_action_detail_se_wrmean.RData")
sessions_device_type_se_wrmean <- readRDS("cache/sessions_device_type_se_wrmean.RData")
sessions_action_se_wmean <- readRDS("cache/sessions_action_se_wmean.RData")
sessions_action_type_se_wmean <- readRDS("cache/sessions_action_type_se_wmean.RData")
sessions_action_detail_se_wmean <- readRDS("cache/sessions_action_detail_se_wmean.RData")
sessions_device_type_se_wmean <- readRDS("cache/sessions_device_type_se_wmean.RData")


# **************************************
# countries features
# **************************************
countries <- dplyr::mutate(countries,
                           country_language = paste0(country_destination, "_", destination_language))
countries_reshape <- data.frame()
for(i in unique(countries$country_language)){
  # i <- "AU_eng"
  countries_ <- subset(countries, country_language == i)
  countries_$country_language <- NULL
  countries_ <- reshape(countries_,
                        direction='wide',
                        idvar='destination_language',
                        timevar='country_destination')
  countries_reshape <- bind_rows(
    countries_reshape,
    countries_
  )
}

countries_reshape <- countries_reshape %>%
  dplyr::group_by(destination_language) %>%
  dplyr::summarise_each(funs(Sum))

countries_reshape <- dplyr::mutate(countries_reshape,
                                   destination_language = str_sub(destination_language, 1, 2))

df_all_countries_feats <- dplyr::left_join(df_all[c("id", "language")],
                                           countries_reshape,
                                           by = c("language" = "destination_language"))
df_all_countries_feats$language <- NULL
df_all_countries_feats <- melt.data.table(as.data.table(df_all_countries_feats))
df_all_countries_feats <- data.frame(df_all_countries_feats)
names(df_all_countries_feats) <- c("id", "feature", "value")
print("countries feature")
print(n_distinct(df_all_countries_feats$feature))


# **************************************
# age_gender_bkts features
# **************************************
age_gender_bkts <- age_gender_bkts %>%
  dplyr::left_join(., countries, by = "country_destination")

age_gender_bkts <- age_gender_bkts[c("age_bucket", "country_destination", "gender", "population_in_thousands", "year", "destination_language")]

age_gender_bkts <- dplyr::mutate(age_gender_bkts,
                                 country_language = paste0(country_destination, "_", destination_language))
age_gender_bkts_reshape <- data.frame()
for(i in unique(age_gender_bkts$country_language)){
  # i <- "AU_eng"
  age_gender_bkts_ <- subset(age_gender_bkts, country_language == i)
  age_gender_bkts_$country_language <- NULL
  age_gender_bkts_ <- reshape(age_gender_bkts_,
                              direction='wide',
                              idvar=c('destination_language', 'age_bucket', 'gender', 'year'),
                              timevar='country_destination')
  age_gender_bkts_reshape <- bind_rows(
    age_gender_bkts_reshape,
    age_gender_bkts_
  )
}

age_gender_bkts_reshape$year <- NULL
age_gender_bkts_reshape <- age_gender_bkts_reshape %>%
  dplyr::mutate(.,
                gender = toupper(gender),
                language = str_sub(destination_language, 1, 2)
  ) %>%
  dplyr::select(-destination_language) %>%
  dplyr::group_by(age_bucket, gender, language) %>%
  dplyr::summarise_each(funs(Sum))

df_all_age_gender_bkts_feats <- dplyr::left_join(df_all[c("id", "age_bucket", "gender", "language")],
                                                 age_gender_bkts_reshape,
                                                 by = c("age_bucket", "gender", "language"))
df_all_age_gender_bkts_feats$age_bucket <- NULL
df_all_age_gender_bkts_feats$gender <- NULL
df_all_age_gender_bkts_feats$language <- NULL
df_all_age_gender_bkts_feats <- melt.data.table(as.data.table(df_all_age_gender_bkts_feats))
df_all_age_gender_bkts_feats <- data.frame(df_all_age_gender_bkts_feats)
names(df_all_age_gender_bkts_feats) <- c("id", "feature", "value")
print("age_gender_bkts feature")
print(n_distinct(df_all_age_gender_bkts_feats$feature))


# **************************************
# feature binding
# **************************************
df_all_feats <- 
  bind_rows(
    df_all_num_feats,
    df_all_ohe_feats,
    sessions_action_se_sum,
    sessions_action_type_se_sum,
    sessions_action_detail_se_sum,
    sessions_device_type_se_sum,
    sessions_action_flg_sum,
    sessions_action_type_flg_sum,
    sessions_action_detail_flg_sum,
    sessions_device_type_flg_sum,
    # sessions_action_se_mean,
    # sessions_action_type_se_mean,
    # sessions_action_detail_se_mean,
    # sessions_device_type_se_mean,
    # sessions_action_se_sd,
    # sessions_action_type_se_sd,
    # sessions_action_detail_se_sd,
    # sessions_device_type_se_sd,
    # sessions_action_se_wrmean,
    # sessions_action_type_se_wrmean,
    # sessions_action_detail_se_wrmean,
    # sessions_device_type_se_wrmean,
    # sessions_action_se_wmean,
    # sessions_action_type_se_wmean,
    # sessions_action_detail_se_wmean,
    # sessions_device_type_se_wmean,
    df_all_countries_feats,
    df_all_age_gender_bkts_feats
  )
print("feature number")
print(n_distinct(df_all_feats$feature))

saveRDS(labels, paste0("cache/",folder,"/labels.RData"))
saveRDS(sample_submission_NDF, paste0("cache/",folder,"/sample_submission_NDF.RData"))
saveRDS(df_all, paste0("cache/",folder,"/df_all.RData"))
saveRDS(df_all_feats, paste0("cache/",folder,"/df_all_feats.RData"))