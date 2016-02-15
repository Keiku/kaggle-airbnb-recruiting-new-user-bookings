# set your working directory
setwd("kaggle-airbnb-recruiting-new-user-bookings")

# **************************************
# create directory
# **************************************
dir.create(paste0("data")) # Please put the download data into this folder
dir.create(paste0("cache"))
dir.create(paste0("submit"))

# set a working folder
folder <- "analysis01"

dir.create(paste0("cache/", folder))
dir.create(paste0("cache/", folder, "/valid"))
dir.create(paste0("cache/", folder, "/test"))
dir.create(paste0("submit/", folder))

# **************************************
# run scripts
# **************************************
source("script/utils.R")
source("script/preprocessing.R")
# stacking.R takes long time (about 2,3 days).
source("script/stacking.R")
# rfs_xgboost.R takes long time (about 2,3 days).
source("script/rfs_xgboost.R")
source("script/submission.R")