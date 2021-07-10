
# HR Analytics: Job Change of Data Scientists -----------------------------


# Required libraries ------------------------------------------------------

library(data.table)
library(ggplot2)
library(caret)
library(mlr)
library(SmartEDA)
library(cowplot)


# Load data set -----------------------------------------------------------
getwd() ## check the current work directory
setwd("C:/backup/StudyMaterials/SDM Workshop/churn_model") ## change working directory as needed
csv_drir <- dir(pattern = ".csv") ## to check the list of items in current wd
hrdata <- fread("hr_trian.csv")
dim(hrdata)
View(head(hrdata))
summary(hrdata) ## default R option to check what data

# Quick analysis on data --------------------------------------------------
## checking the missing data loaded correctly
sapply(hrdata, function(x) length(x[is.na(x)]))
setDF(hrdata)
setDT(hrdata)
class(hrdata)

## Replace all blanks by NA values
hrdata_1 = lapply(hrdata, function(x) ifelse(x=='', NA, x))
setDT(hrdata_1)
View(head(hrdata_1))

rm(hrdata)

## Check column or variable names
names(hrdata_1) ## colnames(hrdata_1)
mycol <- names(hrdata_1) ## define names

## check column type - What type of data captured in each column
sapply(hrdata_1, class)

## Why SmartEDA
View(ExpData(hrdata_1))
View(ExpData(hrdata_1, type=2))


# 1. Dealing with Missing Data using R ------------------------------------

# The following command gives the sum of missing values in the whole data frame column wise
sum(is.na(hrdata_1$experience)) ## to check the independent column missing values
sum(is.na(hrdata_1$gender))

colSums(is.na(hrdata_1))  ## to check overal DF missing value counts

## 1. Missing values can be treated using following methods :
## Deletion
## Mean/ Mode/ Median Imputation
## Prediction Model
## KNN Imputation

## List of R Packages

## a) MICE and VIM
# PMM (Predictive Mean Matching) — For numeric variables
# logreg(Logistic Regression) — For Binary Variables( with 2 levels)
# polyreg(Bayesian polytomous regression) — For Factor Variables (>= 2 levels)
# Proportional odds model (ordered, >= 2 levels)
# VIM is to visualizing the missing values
library(mice)
library(VIM)
md.pattern(hrdata_1)
hrdata_1$enrollee_id <- NULL
mice_plot <- aggr(hrdata_1, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(hrdata_1)[3], cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

imputed_Data <- mice(hrdata_1, m=5, maxit = 50, method = 'pmm', seed = 500)
# Since there are 5 imputed data sets, you can select any using complete() function.
completeData <- complete(imputed_Data,2)

## b) Amelia
library(Amelia)
amelia_fit <- amelia(hrdata_1, m=5, parallel = "multicore")

## c) missForest
library(missForest)

## d) Hmisc
library(Hmisc)

## e) mi
library(mi)

## Impute using direct method mean/median/mode
colSums(is.na(hrdata_1))
## Impute variable by variable
setDF(hrdata_1)
df_imputed = impute(hrdata_1, target = character(0),
                    cols = list(experience = 'xyx',
                                enrolled_university = imputeMode(),
                                education_level = 'abc',
                                major_discipline = 'missing',
                                gender = imputeMode()))

## To faster impuatation for large amount of variable use below method
## Define impute method for each list of variable type
myimpute <- list(factor = imputeMode(),
                              integer = imputeMean(),
                              numeric = imputeMedian(),
                              character = imputeMode())
setDF(hrdata_1)
df_imputed <- impute(hrdata_1, target = character(0), classes = myimpute)
df_imputed$desc

## Create master imputed data set
df_imputed <- df_imputed$data
View(head(df_imputed))

## One quick check on missing values
colSums(is.na(df_imputed))
print("My data is ready for EDA")


# EDA ---------------------------------------------------------------------

## Lets Define columns buckets
pkey_col <- mycol[sapply(hrdata_1, function(x) length(unique(x))==nrow(hrdata_1))]
numeric_col <- mycol[sapply(hrdata_1, function(x) is.integer(x)|is.numeric(x))]
text_col <- mycol[sapply(hrdata_1, function(x) is.factor(x)|is.character(x))]

## What is my target variable
table(df_imputed$target_var)
prop.table(table(df_imputed$target_var))

## Can we use some automated R packages instead of writing lenghty R packages
## SmartEDA has some functionality
df_imputed$enrollee_id = as.character(df_imputed$enrollee_id)
SmartEDA::ExpReport(df_imputed, Target = "target_var", op_file = "hr_eda.html", Rc=1)


## Lets look at two plots for same variable
num_1 <- ExpTwoPlots(df_imputed,
                      plot_type = "numeric",
                      iv_variables = c("city_development_index","training_hours"),
                      target = "target_var",
                      lp_arg_list = list(fill = "grey"),
                      lp_geom_type = 'density',
                      rp_arg_list = list(fill = c("blue", "orange"), alpha=0.5),
                      rp_geom_type = 'boxplot',
                      page = c(2,1),
                      theme = "Default")


# Histogram and Boxplots for numeric variables
box_theme<- theme(axis.line=element_blank(),
                  axis.title=element_blank(),
                  axis.ticks=element_blank(),
                  axis.text=element_blank())

box_theme_y<- theme(axis.line.y=element_blank(),
                    axis.title.y=element_blank(),
                    axis.ticks.y=element_blank(),
                    axis.text.y=element_blank(),
                    legend.position="none")

plot_grid(ggplot(df_imputed, aes(training_hours))+
            geom_histogram(binwidth = 10),
          ggplot(df_imputed, aes(x="",y=training_hours))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, align = "v",ncol = 1)

plot_grid(ggplot(df_imputed, aes(city_development_index))+
            geom_histogram(binwidth = 10),
          ggplot(df_imputed, aes(x="",y=city_development_index))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, align = "v",ncol = 1)

# Data Preparations -------------------------------------------------------

#####Outlier treatment
# Remove outlier
# Treat outlier using imputation, cap, or flag

box <- boxplot.stats(df_imputed$training_hours)
out <- box$out
length(out)
ad1 <- hrdata[ !hrdata$YearsAtCompany %in% out, ]

## Using SmartEDA
out_treat <- ExpOutliers(df_imputed,
            varlist = c("training_hours","city_development_index"),
            method = 'BoxPlot',
            treatment = "mean",
            capping = c(0.05, 0.95),
            outflag = TRUE)
View(out_treat$outlier_summary)

df_new <- out_treat$outlier_data
View(head(df_new))
mynams <- colnames(df_new)
setDF(df_new)

#Feature scaling (Not mandatory - developer call)
num_var<-c("training_hours","city_development_index")
df_new[num_var] <- lapply(df_new[num_var], scale)
df_new$city_development_index <- as.numeric(df_new$city_development_index)
df_new$training_hours <- as.numeric(df_new$training_hours)

# check target variable type
class(df_new$target_var)
table(df_new$target_var)

#Creating categorical subset from the dataset
Cat_var <- mynams[sapply(df_new, function(x) is.factor(x)|is.character(x))]
Cat_var <- setdiff(Cat_var, "enrollee_id")

sapply(df_new[, Cat_var], class)

# converting categorical attributes to factor
class(df_new)
setDT(df_new)
class(df_new$gender)
df_new$gender <- as.factor(df_new$gender)
df_new$relevent_experience  <- as.factor(df_new$relevent_experience )
df_new$experience <- as.factor(df_new$experience)

df_new[, (Cat_var) := lapply(.SD, as.factor), .SDcols = Cat_var]
str(df_new)

# Create dummy variables - onehot encoding
myDummy <- dummyVars(as.formula(paste0("~",paste(Cat_var,collapse = "+"))),df_new)
sparseMtx <- predict(myDummy, df_new)
print(dim(sparseMtx))
colnames(sparseMtx)
colnames (sparseMtx) <- gsub(" ", "_", colnames (sparseMtx))
colnames (sparseMtx) <- gsub("<", "lt", colnames (sparseMtx))
colnames (sparseMtx) <- gsub(">", "gt", colnames (sparseMtx))
colnames (sparseMtx) <- gsub("-", "to", colnames (sparseMtx))
colnames (sparseMtx) <- gsub("\\.", "_", colnames (sparseMtx))
colnames (sparseMtx) <- gsub("\\+", "plus", colnames (sparseMtx))
colnames (sparseMtx) <- gsub(",", "", colnames (sparseMtx))
colnames (sparseMtx) <- gsub ("=", "", colnames (sparseMtx))
colnames(sparseMtx)
class(sparseMtx)
sparseMtx <- as.data.frame (sparseMtx)
setDT (sparseMtx)
View(head(sparseMtx))
## Replace missing values in dummy data matrix
sparseMtx[is.na(sparseMtx)] <- 0
dummy_variable <- names(sparseMtx)
print(paste("Total derived dummy variables", length(dummy_variable)))

## All numeric variables
num_col <- names(df_new)[sapply(df_new, function(x) is.integer(x)|is.numeric(x))]
num_col
# Remove target_var from numeric list
num_col <- setdiff(num_col, "target_var")
print(paste("Total derived numeric variables", length(num_col)))

### Final variable list
myvariable <- unique(c(num_col, dummy_variable))
print(paste("Total independent variables", length(myvariable)))
# 13 --> 192

### Final model data
master_data <- cbind.data.frame(df_new, sparseMtx)
write.csv(master_data,"master_hr_data.csv") ## into csv
saveRDS(master_data, "master_hr_data.rds")
save.image(file="data_preparation_process_v1.rdata")
gc()

# Variable reduction ------------------------------------------------------
# Identify Zero variance variables
setDT(master_data)
myNZV <- nearZeroVar(as.data.frame (master_data[,myvariable,with=F]), saveMetrics = TRUE)
zeroNms <- myvariable [myNZV$zeroVar]
print(paste("Total zero variance columns ", length(zeroNms)))
if(length(zeroNms)>0)
  myvariable <- setdiff(myvariable, zeroNms)

# Identify correlated variables
myDF <- cor(as.data.frame (master_data[,myvariable,with=F]))

myDF <- as.matrix (myDF)
myDF <- ifelse (is.na(myDF), 0, myDF)
myCor <- findCorrelation(myDF, cutoff = 0.95)
corNms <- myvariable [myCor]
print(paste("Total highly correlated columns ", length(corNms)))
if(length(corNms)>0)
  myvariable <- setdiff(myvariable, corNms)

# Low area under curve variables removal
calc_auc <- function (actual, predicted)
{
  r <- rank(predicted)
  n_pos <- as.numeric (sum(actual == 1))
  n_neg <- as.numeric (length(actual) - n_pos)
  denom <- as.double (as.double (n_pos) * as.double(n_neg))
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(denom)
  auc
}

myTgt <- "target_var"

actual <- as.data.frame (master_data)[,myTgt]
table(actual)

aucDF <- master_data [,myvariable,with=F][,lapply(.SD, function (x) calc_auc (actual, x))]
aucDF <- as.data.frame (aucDF)
aucDF <- t (aucDF)
aucDF <- as.data.frame (aucDF)
aucDF$varName <- rownames (aucDF)
names (aucDF)[1] <- "auc"
aucDF <- aucDF [order(aucDF$auc, decreasing=TRUE),]
aucDF$imp <- abs (aucDF$auc - 0.5)
rownames (aucDF) <- NULL
aucDF <- aucDF [order(aucDF$imp, decreasing=TRUE),]
View(head(aucDF))
write.csv (aucDF, file = 'aucDF_SFDC.csv', row.names = F)
rownames (aucDF) <- NULL

varSel_auc <- aucDF$varName [which (aucDF$imp > 0.0008)]

## Final check before Model building
sum (master_data [,myvariable,with=F][,lapply(.SD, function (x) length (which(is.na(x)))/length(x))])
sum (master_data [,myvariable,with=F][,lapply(.SD, function (x) length (which((x==Inf)))/length(x))])
sum (master_data [,myvariable,with=F][,lapply(.SD, function (x) length (which((x==-Inf)))/length(x))])

print("Data prep completed")
save.image(file="data_preparation_process_v1.rdata")
save(varSel_auc, file="AUC_varlist.rdata")


## lets clear R environment to increase the RAM space
rm(list = ls())
gc()
library(data.table)
library(caret)
library(mlr)
library(Metrics)

dir()
