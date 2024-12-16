library(TSA)
library(astsa)
library(forecast)
library(tseries)
library(ggplot2)

# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

# Load dataset
data <- read.csv("TimeSeries_TotalSolarGen_and_Load_IT_2016.csv")

# Convert timestamp to datetime
data$utc_timestamp <- as.POSIXct(data$utc_timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

# Handle missing values (impute with mean)
data$IT_load_new[is.na(data$IT_load_new)] <- mean(data$IT_load_new, na.rm = TRUE)

# Check for missing values
cat("Number of missing values in IT_load_new:", sum(is.na(data$IT_load_new)), "\n")

# Create a time series object for IT_load_new
load_ts <- ts(data$IT_load_new, frequency = 24, start = c(2016, 1)) # Hourly data


# ---------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------
# Plot Load and Solar Generation
ggplot(data, aes(x = utc_timestamp)) +
  geom_line(aes(y = IT_load_new, color = "Load")) +
  geom_line(aes(y = IT_solar_generation, color = "Solar Generation")) +
  labs(title = "Load and Solar Generation Over Time", x = "Time", y = "Value") +
  theme_minimal() +
  scale_color_manual(name = "Legend", values = c("Load" = "blue", "Solar Generation" = "orange"))


# ---------------------------------------------------------------------
# STATIONARITY CHECK AND DIFFERENCING
# ---------------------------------------------------------------------

# Perform Augmented Dickey-Fuller Test
adf_test <- adf.test(load_ts, alternative = "stationary")
print(adf_test)

# Examine ACF, PACF & EACF
acf(load_ts, main = "ACF of Differenced")
pacf(load_ts, main = "PACF of Differenced")
eacf_result <- eacf(load_ts)


# ---------------------------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------------------------

# Split data into training and test sets
train_size <- floor(0.8 * length(load_ts))  

train <- window(load_ts, end = c(2016, train_size))  
test <- window(load_ts, start = c(2016, train_size + 1)) 

cat("Training set length:", length(train), "\n")
cat("Test set length:", length(test), "\n")


# ---------------------------------------------------------------------
# FITTING MODELS
# ---------------------------------------------------------------------

# 1. Fit ARIMA Model (auto.arima) on training set
arima_auto <- auto.arima(train, seasonal = TRUE)
arima_auto

# 2. Fit ARIMA(2,0,2) (based on ACF/PACF)
arima_202 <- Arima(train, order = c(2, 0, 2), method='ML')
arima_202


# ---------------------------------------------------------------------
# RESIDUAL DIAGNOSTICS
# ---------------------------------------------------------------------

# Diagnostics for auto.arima
tsdiag(arima_auto)
acf(residuals(arima_auto), main = "ACF of Residuals (auto.arima)")
qqnorm(residuals(arima_auto))
qqline(residuals(arima_auto))

# Diagnostics for ARIMA(2,0,2)
tsdiag(arima_202)
acf(residuals(arima_202), main = "ACF of Residuals (ARIMA(2,0,2))")
qqnorm(residuals(arima_202))
qqline(residuals(arima_202))


# ---------------------------------------------------------------------
# FORECASTING
# ---------------------------------------------------------------------

# Forecast on test set using each model
# 1. auto.arima
sarima.for(load_ts,48,4,0,3)
forecast_auto <- forecast(arima_auto, h = length(test))
plot(forecast_auto, main = "7-Day Forecast (auto.arima)", xlab = "Time", ylab = "Load")

# 2. ARIMA(2,0,2)
sarima.for(load_ts,48,2,0,2)
forecast_202 <- forecast(arima_202, h = length(test))
plot(forecast_202, main = "Forecast (ARIMA(2,0,2))", xlab = "Time", ylab = "Load")



# ---------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------

# Compare AIC and BIC
cat("auto.arima: AIC =", AIC(arima_auto), ", BIC =", BIC(arima_auto), "\n")
cat("ARIMA(2,0,2): AIC =", AIC(arima_202), ", BIC =", BIC(arima_202), "\n")


# Accuracy on test set
accuracy_auto <- accuracy(forecast_auto, test)
accuracy_202 <- accuracy(forecast_202, test)


cat("Forecast Accuracy (RMSE and MAPE):\n")
cat("auto.arima:\n")
print(accuracy_auto)
cat("ARIMA(2,0,2):\n")
print(accuracy_202)

