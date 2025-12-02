setwd("C:/Path/To/Your/Directory")

library(tidyverse)
library(xts)       
library(tseries)   
library(forecast)  
library(car)
library(lmtest)

data.df <- read.csv("Project Data.csv")

data.df$Date <- as.Date(data.df$Date, format = "%d-%m-%y")

print(head(data.df))
print(str(data.df))

data_prepared <- data.df %>%
  select(Date, NVDA_Close.Adj, VXN_Close, CPI_Day) %>% 
  mutate(
    log_close = log(NVDA_Close.Adj), 
    log_vxn_close = log(VXN_Close),    
    cpi = CPI_Day                       
  ) %>%
  arrange(Date) 

ts_y <- xts(data_prepared$log_close, order.by = data_prepared$Date)
colnames(ts_y) <- "log_close"
ts_x1_vxn <- xts(data_prepared$log_vxn_close, order.by = data_prepared$Date)
ts_x2_cpi <- xts(data_prepared$cpi, order.by = data_prepared$Date)

par(mfrow = c(3, 1), mar = c(3, 4, 3, 2))
plot(ts_y, main = "Endogenous Variable (Y): Log(Nvidia Close)", 
     ylab = "Log(Price)", col = "black")
plot(ts_x1_vxn, main = "Exogenous Variable (X1): Log(VXN Index)", 
     ylab = "Log(Index)", col = "darkblue")
plot(ts_x2_cpi, main = "Exogenous Variable (X2): CPI Day", 
     ylab = "Event (0/1)", col = "red", type = "h")
par(mfrow = c(1, 1))

split_point <- floor(0.8 * nrow(ts_y))
y_train <- ts_y[1:split_point, ]
y_test <- ts_y[(split_point + 1):nrow(ts_y), ]

print(paste("Training set size:", nrow(y_train)))
print(paste("Test set size:", nrow(y_test)))

adf.test(y_train)
kpss.test(y_train)

y_train_diff1 <- diff(y_train, differences = 1) %>% na.omit()

plot(y_train_diff1, main = "NVIDIA Log-Returns (d=1)", 
     ylab = "Log-Return")

adf.test(y_train_diff1)
kpss.test(y_train_diff1)

par(mfrow = c(1, 2))
acf(y_train_diff1, main = "ACF of Stationary Log-Returns (d=1)")
pacf(y_train_diff1, main = "PACF of Stationary Log-Returns (d=1)")
par(mfrow = c(1, 1))

model_010 <- Arima(y_train, order = c(0, 1, 0), include.drift = TRUE) # Random Walk
model_110 <- Arima(y_train, order = c(1, 1, 0), include.drift = TRUE)
model_011 <- Arima(y_train, order = c(0, 1, 1), include.drift = TRUE)
model_111 <- Arima(y_train, order = c(1, 1, 1), include.drift = TRUE)
model_211 <- Arima(y_train, order = c(2, 1, 1), include.drift = TRUE)

data.frame(
  Model = c("ARIMA(0,1,0)", "ARIMA(1,1,0)", "ARIMA(0,1,1)", "ARIMA(1,1,1)", "ARIMA(2,1,1)"),
  AIC = c(AIC(model_010), AIC(model_110), AIC(model_011), AIC(model_111), AIC(model_211))
) %>% arrange(AIC)

best_arima_model <- model_010

checkresiduals(best_arima_model)

print(best_arima_model)

data_prepared_x <- data_prepared %>%
  mutate(
    x1_diff = log_vxn_close - lag(log_vxn_close),
    x2_cpi = cpi
  )

data_prepared_xreg <- data_prepared_x %>%
  mutate(
    x1_lag1 = lag(x1_diff, 1),
    x2_lag1 = lag(x2_cpi, 1)
  ) %>%
  na.omit()

xreg_all <- as.matrix(data_prepared_xreg[, c("x1_lag1", "x2_lag1")])

y_arimax_all_xts <- xts(data_prepared_xreg$log_close, 
                        order.by = data_prepared_xreg$Date)

split_ax <- floor(0.8 * nrow(xreg_all))

y_train_ax <- y_arimax_all_xts[1:split_ax, ]
y_test_ax <- y_arimax_all_xts[(split_ax + 1):nrow(xreg_all), ]

xreg_train <- xreg_all[1:split_ax, ]
xreg_test <- xreg_all[(split_ax + 1):nrow(xreg_all), ]

print(paste("Y Train (ARIMAX):", length(y_train_ax)))
print(paste("Xreg Train (ARIMAX):", nrow(xreg_train)))

vif_data <- as.data.frame(xreg_train)

dummy_lm_model <- lm(as.numeric(y_train_ax) ~ ., data = vif_data)

vif_values <- vif(dummy_lm_model)

print("Variance Inflation Factor (VIF)")
print(vif_values)

best_arimax_model <- Arima(y_train_ax, 
                           order = c(0, 1, 0), include.drift = TRUE
                           , xreg = xreg_train)

print(best_arimax_model)

significance_test <- coeftest(best_arimax_model)

print(significance_test)

checkresiduals(best_arimax_model)

h_steps <- nrow(y_test)
h_steps_ax <- nrow(xreg_test) 

fc_arima <- forecast(best_arima_model, h = h_steps)
print(fc_arima)
plot(fc_arima)

fc_arimax <- forecast(best_arimax_model, 
                      h = h_steps_ax, 
                      xreg = xreg_test)
print(fc_arimax)
plot(fc_arimax)

actual_prices_arima <- exp(y_test)
actual_prices_arimax <- exp(y_test_ax)

pred_prices_arima <- exp(fc_arima$mean)
pred_prices_arimax <- exp(fc_arimax$mean)

acc_arima <- accuracy(pred_prices_arima, actual_prices_arima)
acc_arimax <- accuracy(pred_prices_arimax, actual_prices_arimax)

print("ARIMA Accuracy (Test Set)")
print(acc_arima)

print("ARIMAX Accuracy (Test Set)")
print(acc_arimax)

comparison_table <- data.frame(
  Model = c("ARIMA", "ARIMAX"),
  RMSE = c(acc_arima[,"RMSE"], acc_arimax[,"RMSE"]),
  MAE = c(acc_arima[,"MAE"], acc_arimax[,"MAE"]),
  MAPE = c(acc_arima[,"MAPE"], acc_arimax[,"MAPE"])
)
print(comparison_table)



