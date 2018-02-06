# install.packages(c("pROC", "tidyverse"))
# install.packages("e1071")
library(pROC)
library(tidyverse)
library(car)
library(radiant)
library(caret)
library(e1071)
library(randomForest)
# ==========================================read and clean data================================================
#Load data
raw <- read_rds("Data/Humana_Orig") 

condition <- as.data.frame(matrix(nrow = nrow(raw), ncol = 66))

# For loop body : take out Q1-Q4, Q5-Q8 separately based on specific condition and 
# calculate rowsums to get annual total for each year.
for (i in 1:33) {
  if(i < 10) {
    condition[ , i * 2 - 1] <- raw %>% 
      select(matches(paste("^CON_[A-Z]{5}_0", i, "_Q0[0-4]", sep = ""))) %>% 
      rowSums
    condition[ , i * 2] <- raw %>% 
      select(matches(paste("^CON_[A-Z]{5}_0", i, "_Q0[5-8]", sep = ""))) %>% 
      rowSums
  } else { 
    condition[ , i * 2 - 1] <- raw %>% 
      select(matches(paste("^CON_[A-Z]{5}_", i, "_Q0[0-4]", sep = ""))) %>% 
      rowSums
    condition[ , i * 2] <- raw %>% 
      select(matches(paste("^CON_[A-Z]{5}_", i, "_Q0[5-8]", sep = ""))) %>% 
      rowSums
  }
  colnames(condition)[(i * 2 - 1):(i * 2)] <- paste0(paste("Condi_Code_", i, sep = ""), c("_in_2014", "_in_2015"))
}

#Replacing original "CON_VISIT_x_Qx" with aggregated columns (in "condition"),
#both colnames and values
raw[ , grep(x = names(raw), pattern = "CON_VISIT_*")] <- condition

CON_IND <- grep(x = names(raw), pattern = "CON_VISIT_*")

colnames(raw)[CON_IND[1:66]] <- colnames(condition)
raw_aggr_con_2years <- raw[ , -grep(x = names(raw), pattern = "CON_VISIT_*")]

# Do the same transformation with place visit data 

place <- as.data.frame(matrix(nrow = nrow(raw_aggr_con_2years), ncol = 96))
j <- 0

for (i in c(1:9, 11:26, 31:34, 41:42, 49:57, 60:62, 65, 71:72, 81, 99)) {
  j <- j + 1
  if(i < 10) {
    place[ , j * 2 - 1] <- raw_aggr_con_2years %>% 
      select(matches(paste("^POT_[A-Z]{5}_0", i, "_Q0[0-4]", sep = ""))) %>% 
      rowSums
    place[ , j * 2] <- raw_aggr_con_2years %>% 
      select(matches(paste("^POT_[A-Z]{5}_0", i, "_Q0[5-8]", sep = ""))) %>% 
      rowSums
    
  } else { 
    place[ , j * 2 - 1] <- raw_aggr_con_2years %>% 
      select(matches(paste("^POT_[A-Z]{5}_", i, "_Q0[0-4]", sep = ""))) %>% 
      rowSums
    place[ , j * 2] <- raw_aggr_con_2years %>% 
      select(matches(paste("^POT_[A-Z]{5}_", i, "_Q0[5-8]", sep = ""))) %>% 
      rowSums
  }
  colnames(place)[(j * 2 - 1):(j * 2)] <- paste0(paste("Place_Code_", i, sep = ""), c("_in_2014", "_in_2015"))
}



raw_aggr_con_2years[ , grep(x = names(raw_aggr_con_2years), pattern = "POT_VISIT_*")] <- place

POT_IND <- grep(x = names(raw_aggr_con_2years), pattern = "POT_VISIT_*")

colnames(raw_aggr_con_2years)[POT_IND[1:96]] <- colnames(place)
main_aggr_con_pot <- raw_aggr_con_2years[ , -grep(x = names(raw_aggr_con_2years), pattern = "POT_VISIT_*")]


main <- main_aggr_con_pot %>%
  select(-ends_with("2016"))

# change variables'type of character to categorical data
main[which(lapply(main, class) == "character")] <-
  main[which(lapply(main, class) == "character")] %>%
  purrr::map_df(as.factor)

# fill NA of missing value
fill_fun <- function(x) {
  b <- is.na(x)
  x[b] <- sample(x[!b], sum(b), replace = TRUE)
  x
}
main <- purrr::map_df(main, fill_fun)


# remove non-varied varaibles
non_var = c()
for (i in 1:length(main)) {
  if (length(table(main[, i])) == 1) {
    non_var <- c(non_var, i)
  }
}
main <- main[, -non_var]

main$ADMISSIONS <- as.numeric(main$ADMISSIONS != 0)
main$READMISSIONS <- as.numeric(main$READMISSIONS != 0)

# divide to test dataset and training dataset

train_sub <- sample(nrow(main), 7 / 10 * nrow(main))
train <- main[train_sub, ]
test <- main[-train_sub, ]

# ============================================select variables=================================================
# Step1 rough selection based on correlation
corr <- function(x) {
  cor.test(as.vector(t(x)), dependent, method = "pearson")[["estimate"]]
}

train_nofct <- train[which(lapply(train, class) != "factor")] %>%
  purrr::map_df(as.double)

dependent <- as.double(train_nofct$ADMISSIONS)

#Do correlation test (corr >= 0.05)
test_corr <- as.data.frame(t(purrr::map_df(train_nofct, corr))) %>%
  data.frame("variables" = rownames(.))


    #Variables that passed test
selected_vars <- filter(test_corr, abs(V1) >= 0.05) %>%
  select(variables, V1)


train_corr <- select(train, selected_vars$variables) %>%
  cbind(train[which(lapply(train, class) == "factor")])

train_predictor <- names(train_corr[,-c(1,2)])


# Step2. Further selection using regression model stats
result <- logistic(dataset = train_corr, rvar = "ADMISSIONS", evar = train_predictor,lev = "1", check = "standardize")

LR_var <- result$coeff[, c(1, 3, 6)]
colnames(LR_var)[1] <-  "variables"

LR_delete <- LR_var %>% 
  filter(abs(coefficient) <= 0.05 & p.value >= 0.05)

LR_keep <- LR_var %>% 
  filter(!(LR_var$variables %in% LR_delete$variables))

train_selected <- train_corr[which(colnames(train_corr) %in% LR_keep$variables)] %>%
  cbind(train_corr[, c(
    "ESRD_IND",
    "HOSPICE_IND",
    "PCP_ASSIGNMENT",
    "Dwelling_Type",
    "LIS",
    "INSTITUTIONAL",
    "MAJOR_GEOGRAPHY",
    "MINOR_GEOGRAPHY",
    "MCO_PROD_TYPE_CD"
  )],
  ADMISSIONS = train_corr$ADMISSIONS)

train_selected_predictor <- names(train_selected)[-length(train_selected)]

saveRDS(train_selected,"train_selected.rds")
rm(list = c("i","test_corr","train_sub","dependent","non_var", "result", "LR_var", "LR_delete", "LR_keep", "selected_vars"))
# =====================================Logistic Regression Prediction==========================================

LR <- glm(ADMISSIONS ~ ., data = train_selected, family = "binomial")

# predict
prob <- predict(LR,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob >= i)
  t <- confusionMatrix(test$ADMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_LR <- as.numeric(prob >= im)
confusionMatrix(test$ADMISSIONS, prob_LR)
rm(list = c("i", "t", "s", "sm", "prob", "probx"))

# ===============================================ANN prediction================================================

Ann <- ann(dataset = train_selected, rvar = "ADMISSIONS", evar = train_selected_predictor, lev = "1", seed = 1234)

#preict
prob <- predict(Ann,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob$Prediction >= i)
  t <- confusionMatrix(test$ADMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_ANN <- as.numeric(prob$Prediction >= im)
confusionMatrix(test$ADMISSIONS, prob_ANN)
rm(list = c("i", "t", "s", "sm", "prob", "probx"))

# =========================================Random Forest Prediction===========================================
RF <- randomForest(ADMISSIONS ~ ., data=train_selected,importance=TRUE) 

importance(RF,type=2)
varImpPlot(RF)
plot(RF)

prob <- predict(RF,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob >= i)
  t <- confusionMatrix(test$ADMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_RF <- as.numeric(prob >= im)
confusionMatrix(test$ADMISSIONS, prob_RF)
rm(list = c("i", "t", "s", "sm", "prob", "probx"))


# =============================================================================================================
# ==================================Repeat above work process with READMISSIONS================================

# ============================================select variables=================================================
# Step1 rough selection based on correlation
dependent <- as.double(train_nofct$READMISSIONS)

#Do correlation test (corr >= 0.05)
test_corr <- as.data.frame(t(purrr::map_df(train_nofct, corr))) %>%
  data.frame("variables" = rownames(.))


#Variables that passed test
selected_vars <- filter(test_corr, abs(V1) >= 0.05) %>%
  select(variables, V1)


train_corr <- select(train, selected_vars$variables) %>%
  cbind(train[which(lapply(train, class) == "factor")])

train_predictor <- names(train_corr[,-c(1,2)])


# Step2. Further selection using regression model stats
result <-
  logistic(dataset = train_corr, rvar = "READMISSIONS", evar = train_predictor,lev = "1", check = "standardize")

LR_var <- result$coeff[, c(1, 3, 6)]
colnames(LR_var)[1] <-  "variables"

LR_delete <- LR_var %>% 
  filter(abs(coefficient) <= 0.05 & p.value >= 0.05)

LR_keep <- LR_var %>% 
  filter(!(LR_var$variables %in% LR_delete$variables))

train_selected <- train_corr[which(colnames(train_corr) %in% LR_keep$variables)] %>%
  cbind(train_corr[, c(
    "ESRD_IND",
    "HOSPICE_IND",
    "PCP_ASSIGNMENT",
    "Dwelling_Type",
    "LIS",
    "INSTITUTIONAL",
    "MAJOR_GEOGRAPHY",
    "MINOR_GEOGRAPHY",
    "MCO_PROD_TYPE_CD"
  )],
  READMISSIONS = train_corr$READMISSIONS)

train_selected_predictor <- names(train_selected)[-length(train_selected)]

saveRDS(train_selected,"train_selected.rds")
rm(list = c("i", "train_sub", "non_var", "result","test_corr", "LR_var", "LR_delete", "LR_keep", "train_nofct", "selected_vars"))
# =====================================Logistic Regression Prediction==========================================

LR <- glm(READMISSIONS ~ ., data = train_selected, family = "binomial")

# predict
prob <- predict(LR,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob >= i)
  t <- confusionMatrix(test$READMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_01 <- as.numeric(prob >= im)
confusionMatrix(test$READMISSIONS, prob_01)
rm(list = c("i", "t", "s", "sm", "prob", "probx", "prob_01"))

# ===============================================ANN prediction================================================

Ann <- ann(dataset = train_selected, rvar = "READMISSIONS", evar = train_selected_predictor, lev = "1", seed = 1234)

#predict
prob <- predict(Ann,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob$Prediction >= i)
  t <- confusionMatrix(test$READMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_01 <- as.numeric(prob$Prediction >= im)
confusionMatrix(test$READMISSIONS, prob_01)
rm(list = c("i", "t", "s", "sm", "prob", "probx", "prob_01"))
summary(LR)

# =========================================Random Forest Prediction===========================================
RF <- randomForest(READMISSIONS ~ ., data=train_selected,importance=TRUE) 

importance(RF,type=2)
varImpPlot(RF)
plot(RF)

prob <- predict(RF,test,type = "response")
# find threshold 'im' that gives maximum accuracy
sm <- 0
for (i in seq(from = 0.3, to = 0.7, by = 0.001)) {
  probx <- as.numeric(prob >= i)
  t <- confusionMatrix(test$READMISSIONS, probx)
  s <- t[["table"]][1, 1] + t[["table"]][2, 2]
  if (s > sm) {
    sm <- s
    im <- i
  }
}

prob_01 <- as.numeric(prob >= im)
confusionMatrix(test$READMISSIONS, prob_01)
rm(list = c("i", "t", "s", "sm", "prob", "probx", "prob_01"))


confusionMatrix(test$ADMISSIONS, prob_LR)
confusionMatrix(test$ADMISSIONS, prob_ANN)
confusionMatrix(test$ADMISSIONS, prob_RF)
