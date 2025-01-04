library(doParallel)
library(caret)
library(rugarch)


# Data Setup and Processing ####################################################

set.seed(123456)

SPX <- read.csv("SPX.csv", header = TRUE)
VIX <- read.csv("VIX.csv", header = TRUE)
Full_ts <- as.data.frame(cbind(SPX, VIX$VIX))
Train_split <- 0.8
L_ts <- 126
L_pred <- 21

Training_ts <- Full_ts[1:(nrow(Full_ts) * Train_split), ]
Training_df <- matrix(NA, nrow = nrow(Training_ts) - L_ts - L_pred, ncol = L_ts + 1)
Training_Y <- numeric(nrow(Training_df))

for (i in 1:nrow(Training_df)) {
  Training_df[i, (1:L_ts)] <- Training_ts[i:(i + L_ts - 1), 2]
  Training_df[i, L_ts + 1] <- Training_ts[i + L_ts - 1, 3]
  Training_Y[i] <- sd(Training_ts[(i + L_ts):(i + L_ts + L_pred - 1), 2]) * sqrt(252)
}

Training_df <- cbind(Training_df, Training_Y)
Training_df <- data.frame(Training_df[sample(nrow(Training_df)), ])
Training_df0 <- Training_df
Training_df <- Training_df[ , -(L_ts + 1)]
Training_Y <- as.numeric(Training_df[ , L_ts + 1])

Testing_ts <- Full_ts[((nrow(Full_ts) * Train_split) + L_ts): nrow(Full_ts), ]
Testing_df <- matrix(NA, nrow = nrow(Testing_ts) - L_ts - L_pred, ncol = L_ts + 1)
Testing_Y <- numeric(nrow(Testing_df))

for (i in 1:nrow(Testing_df)) {
  Testing_df[i, (1:L_ts)] <- Testing_ts[i:(i + L_ts -1), 2]
  Testing_df[i, L_ts + 1] <- Testing_ts[i + L_ts - 1, 3]
  Testing_Y[i] <- sd(Testing_ts[(i + L_ts):(i + L_ts + L_pred - 1), 2]) * sqrt(252)
}

Testing_df <- data.frame(cbind(Testing_df, Testing_Y))
Testing_df0 <- Testing_df
Testing_df <- Testing_df[ , -(L_ts + 1)]



# Random Forest ################################################################

control <- trainControl(method = 'cv',
                        number = 10,
                        search = 'grid',
                        allowParallel = TRUE)

tunegrid <- expand.grid(mtry = seq(from = 10, to = 120, by = 10))

cl <- makePSOCKcluster(20)
registerDoParallel(cl)

Model_rf <- train(Training_Y ~.,
                  data = Training_df,
                  method = "rf",
                  ntree = 500,
                  tuneGrid = tunegrid, 
                  trControl = control,
                  importance = TRUE)

stopCluster(cl)

saveRDS(Model_rf, "Model_rf.rds")



# Gradient Boosting ############################################################

control <- trainControl(method = 'cv',
                        number = 10,
                        search = 'grid',
                        allowParallel = TRUE)

tunegrid <- expand.grid(nrounds = seq(from = 50, to = 500, by = 50), max_depth = 1,
                        eta = seq(from = 0.05, to = 0.30, by = 0.05), gamma = 0,
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)

Model_gb <- train(as.matrix(Training_df[ , 1:L_ts]),
                  Training_df$Training_Y,
                  method = "xgbTree",
                  tuneGrid = tunegrid,
                  trControl = control, 
                  verbose = TRUE)

saveRDS(Model_gb, "Model_gb.rds")



# Final Model ##################################################################

control <- trainControl(method = 'cv',
                        number = 10,
                        search = 'grid',
                        allowParallel = TRUE)

tunegrid <- expand.grid(nrounds = seq(from = 50, to = 500, by = 50), max_depth = 1,
                        eta = seq(from = 0.05, to = 0.30, by = 0.05), gamma = 0,
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)

Model_final <- train(as.matrix(Training_df0[ , 1:(L_ts + 1)]),
                  Training_df0$Training_Y,
                  method = "xgbTree",
                  tuneGrid = tunegrid,
                  trControl = control, 
                  verbose = TRUE)

saveRDS(Model_final, "Model_final.rds")



# Testing ######################################################################

Testing_df2 <- Testing_df[, (L_ts - 20):L_ts]     # Simple Moving Average
Simple <- apply(Testing_df2, 1, sd) * sqrt(252)
sqrt(mean(((Simple - Testing_Y)**2)))
mean(abs(Simple - Testing_Y))
mean(abs(Simple - Testing_Y)/Testing_Y)

Decay <- 0.88     # Exponentially Weighted Moving Average
EWMA <- function(delta, series){
  series = as.numeric(series)
  Leng = length(series)
  Wgts = numeric(Leng)
  Wgts[1] = 1
  for (i in 2:Leng) {Wgts[i] = Wgts[i - 1] * delta}
  Wgts_0 = rev(Wgts)
  Wgts = Wgts_0/sum(Wgts_0)
  W_mean <- as.numeric(crossprod(Wgts, series))
  S1 = (1 - (delta**Leng))/(1 - delta)
  S2 = (1 - (delta**(2*Leng)))/(1 - (delta**2))
  A = S1/((S1**2) - S2)
  B = S1 * A
  EWMA = A * sum((Wgts_0 * (series**2))) - B*(W_mean**2)
  return(EWMA)}
Expon <- sqrt(apply(Testing_df[ , (1:L_ts)], 1, function(x) EWMA(Decay, x)) * 252)
sqrt(mean(((Expon - Testing_Y)**2)))
mean(abs(Expon - Testing_Y))
mean(abs(Expon - Testing_Y)/Testing_Y)

fit_garch <- ugarchfit(spec = ugarchspec(), data = Full_ts$SPX,         # GARCH
                       out.sample = nrow(Full_ts) - nrow(Training_df), solver = 'hybrid')
forc_garch <- ugarchforecast(fit_garch, n.ahead = L_pred, 
                             n.roll = nrow(Full_ts) - nrow(Training_df))
mat_garch <- as.data.frame(t(forc_garch@forecast[["sigmaFor"]]))
GPred <- sqrt(tail(as.vector(rowMeans(mat_garch)), n = nrow(Testing_df)))
sqrt(mean(((GPred - Testing_Y)**2)))
mean(abs(GPred - Testing_Y))
mean(abs(GPred - Testing_Y)/Testing_Y)

VIX_ts <- VIX[6527:7823, 2]/100         # Option-Implied (VIX)
cor(VIX_ts, Testing_Y)**2
sqrt(mean(((VIX_ts - Testing_Y)**2)))
mean(abs(VIX_ts - Testing_Y))
mean(abs(VIX_ts - Testing_Y)/Testing_Y)

Pred_Y_RF <- predict(Model_rf, newdata = as.matrix(Testing_df[ , 1:L_ts]))     # Random Forest
cor(Pred_Y_RF, Testing_Y)**2
sqrt(mean(((Pred_Y_RF - Testing_Y)**2)))
mean(abs(Pred_Y_RF - Testing_Y))
mean(abs(Pred_Y_RF - Testing_Y)/Testing_Y)

Pred_Y_GB <- predict(Model_gb, newdata = as.matrix(Testing_df[ , 1:L_ts]))     # Gradient Boosting
cor(Pred_Y_GB, Testing_Y)**2
sqrt(mean(((Pred_Y_GB - Testing_Y)**2)))
mean(abs(Pred_Y_GB - Testing_Y))
mean(abs(Pred_Y_GB - Testing_Y)/Testing_Y)

Pred_Y_Final <- predict(Model_final, newdata = as.matrix(Testing_df0[ , 1:(L_ts + 1)]))     # Final Model
cor(Pred_Y_Final, Testing_Y)**2
sqrt(mean(((Pred_Y_Final - Testing_Y)**2)))
mean(abs(Pred_Y_Final - Testing_Y))
mean(abs(Pred_Y_Final - Testing_Y)/Testing_Y)



# Variable Importance ##########################################################

Naive_Weight <- numeric(L_ts)
Naive_Weight[(length(Naive_Weight) - L_pred + 1):(length(Naive_Weight))] <- 1/L_pred

RF_import <- as.data.frame(Model_rf$finalModel$importance[ , 1])
names(RF_import) <- "Importance"

GB_import <- varImp(Model_gb)
GB_import <- as.data.frame(GB_import$importance)
GB_import$nms <- row.names(GB_import)
GB_import$nms <- gsub("V", "", as.character(GB_import$nms))
GB_import$nms <- as.numeric(GB_import$nms)
GB_import <- GB_import[order(GB_import$nms, decreasing = FALSE),]
names(GB_import) <- c("Importance", "nms")

Final_import <- varImp(Model_final)
Final_import <- as.data.frame(Final_import$importance)
Final_import$nms <- row.names(Final_import)
Final_import$nms <- gsub("V", "", as.character(Final_import$nms))
Final_import$nms <- as.numeric(Final_import$nms)
Final_import <- Final_import[order(Final_import$nms, decreasing = FALSE),]
names(Final_import) <- c("Importance", "nms")

