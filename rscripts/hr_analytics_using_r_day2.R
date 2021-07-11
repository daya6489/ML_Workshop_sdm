## Library
library(data.table)
library(xgboost)
library(randomForest)
library(caret)
library(ggplot2)
library(Metrics)
library(ROCit)
library(pROC)

dir()
# Data Modeling -----------------------------------------------------------

######## Train and valid split- We will run k-fold cross validation on train and valid would be used to calculate the lift in the end
data1 <- readRDS("master_hr_data.rds")
load("AUC_varlist.rdata")
str(data1)
dim(data1)
class(data1)

set.seed(12345)
train.index <- createDataPartition(data1$target_var, p = .8, list = FALSE)
train <- data1[train.index,]
valid  <- data1[-train.index,]
dim(train)
dim(valid)

prop.table(table(train$target_var))
prop.table(table(valid$target_var))

write.csv(valid,"valid.csv")
write.csv(train,"train.csv")

## Some best practices
modelList <- list ()
modelNms <- NULL
actualList <- list ()
predList <- list ()


k <- 1
myTgt <- "target_var"
modelNum <- k

# Run Logistic Regression Model -------------------------------------------

myGLM <- glm (as.formula(paste (myTgt, "~", paste(varSel_auc, collapse="+"))),
              as.data.frame (train), family="binomial")

save (myGLM, file = "myGLM.rData")

pred_glm <- predict(myGLM, as.data.frame (valid), type = "response")
summary(pred_glm)

actual <- as.data.frame (valid) [,myTgt]
table(actual)
print (auc(actual, pred_glm)) #[1]  0.7763961

modelList [[modelNum]] <- myGLM
modelNms <- c(modelNms, "GLM 1")
actualList[[modelNum]] <- actual
predList[[modelNum]] <- pred_glm


# Stepwise Logistic Regression --------------------------------------------
actual_var <- names(data1)[1:12]


#Initial model
modelNum <- 2
glm_formula1 <- as.formula(paste (myTgt, "~", paste(actual_var, collapse="+")))
glm_formula1
model_1 = glm(glm_formula1, data = train, family = "binomial")
summary(model_1) #AIC: 963.3 ....Residual deviance: 857.3

# Stepwise selection
library("MASS")
model_2<- stepAIC(model_1, direction="both")
summary(model_2)  #AIC: AIC=14326.11

table(valid$city)
valid1 = valid[valid$city!="city_171",]

pred_glm1 <- predict (model_2, as.data.frame (valid1), type = "response")
summary(pred_glm1)
actual <- as.data.frame (valid1) [,myTgt]
print (auc(actual, pred_glm1)) #[1]  0.7749

modelList [[modelNum]] <- model_2
modelNms <- c(modelNms, "GLM Stepwise")
actualList[[modelNum]] <- actual
predList[[modelNum]] <- pred_glm1

### Evaluate Model (Before Thresholding)
# Let's use the probability cutoff of 50%.
test_pred <- factor(ifelse(pred_glm >= 0.50, "Yes", "No"))
table(test_pred)
test_actual <- factor(ifelse(valid$target_var==1,"Yes","No"))
table(test_actual)

conf_mt_lr <- confusionMatrix(table(test_actual, test_pred), positive = "Yes")
conf_mt_lr
## Model summary
conf_mt_lr$byClass
# F1 = 0.39813582
# Balanced Accuracy = 0.68096410

### Evaluate Model (After Thresholding)
library(pROC)
my_roc <- roc(valid$target_var, pred_glm)
best_cutoff <- coords(my_roc, "best", ret = "threshold")
print(paste("Optimal Probability Thresholds with ROC Curve", round(best_cutoff,3)))

test_pred1 <- factor(ifelse(pred_glm >= best_cutoff$threshold, "Yes", "No"))
conf_mt_lr1 <- confusionMatrix(table(test_actual, test_pred1), positive = "Yes")
conf_mt_lr1

## Model summary - before
conf_mt_lr$byClass
## Model summary - after
conf_mt_lr1$byClass

library(ROCit)
roc_empirical <- rocit(score = pred_glm, class = test_actual)
plot(roc_empirical, values = T, legend = F)

## Gain Table
gtable_custom <- gainstable(roc_empirical, ngroup = 10)
plot(gtable_custom, type = 1)


# Random Forest Model -----------------------------------------------------
# Grid Search or Random of RF parameter
# ntree = [100, 200, 500, 1000, 1500, 3000]
my_rf_all <- randomForest(x=train[,varSel_auc],
                      y = as.factor (train$target_var),
                      ntree = 500,
                      nodesize = 5,
                      do.trace = 25)

save(my_rf_all, file = "random_forest_all.rdata")
## Predict
pred_rf_all <- predict (my_rf_all, valid[,varSel_auc], type = "prob" )[,2] ## 500 trees
head(pred_rf_all)
actual <- as.data.frame (valid) [,myTgt]
print (auc(actual, pred_rf_all)) #[1]  0.7815

# Store Model outcome
modelNum <- 3
modelList [[modelNum]] <- my_rf_all
modelNms <- c(modelNms, "RF all")
actualList[[modelNum]] <- actual
predList[[modelNum]] <- pred_rf_all


## Lets run balanced RF model
## =============================================================
table(train$target_var)
my_rf_us <- randomForest(x=train[,varSel_auc],
                         y = as.factor (train$target_var),
                          sampsize = c(2000, 1000),
                          ntree = 100,
                          nodesize = 5,
                          do.trace = 25)

## Predict
pred_rf_us <- predict(my_rf_us, valid[,varSel_auc], type = "prob" )[,2] ## 500 trees
save(pred_rf_us, file="random_forest_us.rdata")
actual <- as.data.frame (valid) [,myTgt]
print (auc(actual, pred_rf_us)) #[1]  0.7894

## Variable importance - RF
var_imp_rf = importance(my_rf_us,type=2)
var_imp_rf = as.data.frame(var_imp_rf)
View(var_imp_rf)

## Model save
modelNum <- 4
modelList [[modelNum]] <- my_rf_us
modelNms <- c(modelNms, "RF US")
actualList[[modelNum]] <- actual
predList[[modelNum]] <- pred_rf_us

## Model validations
## Evaluate Model (Before Thresholding)
# Let's use the probability cutoff of 50%.
test_pred <- factor(ifelse(pred_rf_us >= 0.50, "Yes", "No"))
test_actual <- factor(ifelse(valid$target_var==1,"Yes","No"))

conf_mt_lr_rf <- confusionMatrix(table(test_actual, test_pred), positive = "Yes")
conf_mt_lr_rf
## Model summary
conf_mt_lr_rf$byClass

### Evaluate Model (After Thresholding)
my_roc <- roc(valid$target_var, pred_rf_us)
best_cutoff <- coords(my_roc, "best", ret = "threshold")
print(paste("Optimal Probability Thresholds with ROC Curve", round(best_cutoff,3)))

test_pred1 <- factor(ifelse(pred_rf_us >= best_cutoff$threshold, "Yes", "No"))
conf_mt_lr_rf_1 <- confusionMatrix(table(test_actual, test_pred1), positive = "Yes")
conf_mt_lr_rf_1

## Model summary - before
conf_mt_lr_rf$byClass
## Model summary - after
conf_mt_lr_rf_1$byClass

## Threshold Plot
roc_empirical <- rocit(score = pred_rf_us, class = test_actual)
plot(roc_empirical, values = T, legend = F)

## Gain Table
gtable_custom <- gainstable(roc_empirical, ngroup = 10)
plot(gtable_custom, type = 1)

# XGBoost Model -----------------------------------------------------------

## Convert train and valida data frames inot xgb matrix foramt
class(train)
train_label <- train$target_var
train_xgb <- xgb.DMatrix(data = as.matrix(train[,varSel_auc]), label = train_label)
xgb.DMatrix.save (train_xgb, 'train_xgb.big')

test_label <- valid$target_var
test_xgb <- xgb.DMatrix(data = as.matrix(valid[, varSel_auc]), label = test_label)
xgb.DMatrix.save (test_xgb, 'test_xgb.big')


## set parameters
watchlist <- list (train=train_xgb,validation=test_xgb)

prm <- expand.grid(max_depth = 5,
                   eta = 0.2,
                   min_child_weight = 10,
                   subsample =0.9,
                   colsample_bytree = 0.9,
                   nrounds=1000,
                   nthread = 15)

xgboost_model <- xgb.train(data = train_xgb, nthread = 15,
                           objective = "binary:logistic", nround = prm$nrounds,
                           max_depth = prm$max_depth, eta = prm$eta,
                           min_child_weight = prm$min_child_weight, subsample = prm$subsample,
                           colsample_bytree = prm$colsample_bytree,
                           watchlist = watchlist, eval_metric="logloss",
                           early_stopping_rounds = 30, maximize=TRUE, printEveryN = 20)

## Variable Importance
imp_fearture <- xgb.importance(varSel_auc, model = xgboost_model)
print(imp_fearture)
xgb.plot.importance(imp_fearture)

## Predict
pred_xgb <- predict (xgboost_model, test_xgb, ntreelimit = xgboost_model$bestInd)

## Model AUC
print (auc(actual, pred_xgb)) # 0.7852 ---> .800


### Evaluate Model (Before Thresholding)
# Let's use the probability cutoff of 50%.
test_pred_xgb <- factor(ifelse(pred_xgb >= 0.50, "Yes", "No"))
test_actual <- factor(ifelse(valid$target_var==1,"Yes","No"))

conf_mt_lr <- confusionMatrix(table(test_actual, test_pred_xgb), positive = "Yes")
conf_mt_lr
## Model summary
conf_mt_lr$byClass


### Evaluate Model (After Thresholding)
my_roc <- roc(valid$target_var, pred_xgb)
best_cutoff <- coords(my_roc, "best", ret = "threshold")
print(paste("Optimal Probability Thresholds with ROC Curve", round(best_cutoff,3)))

test_pred_xgb <- factor(ifelse(pred_glm >= best_cutoff$threshold, "Yes", "No"))
conf_mt_lr1 <- confusionMatrix(table(test_actual, test_pred_xgb), positive = "Yes")
conf_mt_lr1

## Model summary - before
conf_mt_lr$byClass
## Model summary - after
conf_mt_lr1$byClass

library(ROCit)
roc_empirical <- rocit(score = pred_xgb, class = test_actual)
plot(roc_empirical, values = T, legend = F)

## Gain Table
gtable_custom <- gainstable(roc_empirical, ngroup = 10)
plot(gtable_custom, type = 1)


