# Load required packages
library(fpp3)
library(readr)
library(tsibble)
library(dplyr)
library(forecast)

x <- 10


data <- read_csv("./data/PJME_hourly.csv",
                 col_types = cols(
                   Datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                   PJME_MW = col_double()
                 ))

data <- data |>
  filter(Datetime >= as.POSIXct("2017-01-01"))


# Check for duplicates
duplicates <- data |> 
  group_by(Datetime) |> 
  filter(n() > 1)

# View the duplicates
print(duplicates)

data_unique <- data |> 
  group_by(Datetime) |> 
  summarise(PJME_MW = mean(PJME_MW, na.rm = TRUE)) |> 
  ungroup()

ts_data <- as_tsibble(data_unique, index = Datetime) |> fill_gaps()

ts_data |>
  fill_gaps() |>
  autoplot() +
  labs(y = "MW",
       title = "PJME Hourly")


fit_pjme <- ts_data |>
  model(TSLM(PJME_MW ~ trend() + season()))


# Calculate the number of hours in a quarter (approximate)
hours_in_quarter <- 24 * 365.25 / 4  # about 2190 hours

# Generate the forecast
fc_pjme <- forecast(fit_pjme, h = hours_in_quarter)


fc_pjme |>
  autoplot(ts_data) +
  labs(
    title = "PJME hourly",
    y = "MW"
  )

######### use ARIMA #################

fit_arima <- ts_data |>
  model(ARIMA(PJME_MW))

fc_arima <- forecast(fit_arima, h = hours_in_quarter)

fc_arima |>
  autoplot(ts_data) +
  labs(
    title = "PJME hourly",
    y = "MW"
  )

######### use ETS #################

fit_ets <- ts_data |>
  model(ets(PJME_MW))

fc_ets <- forecast(fit_ets, h = hours_in_quarter)

fc_ets |>
  autoplot(ts_data) +
  labs(
    title = "PJME hourly",
    y = "MW"
  )

# Print summary of the forecast
summary(fc_ets)


ts_data |> 
  fill_gaps() |>
  slice(-n()) |>
  stretch_tsibble(.init = 10) |>
  model(
    ETS(PJME_MW),
    ARIMA(PJME_MW)
  ) |>
  forecast(h = 72) |>
  select(.model, RMSE:MAPE)


train <- ts_data |> filter_index(. ~ "2018-03-01") |> fill_gaps()

fit_arima <- train |> model(ARIMA(ts_data))
report(fit_arima)

fit_arima |> gg_tsresiduals(lag_max =20)


augment(fit_arima) |> features(.innov, ljung_box, lag = 20, dof = 5)

fit_ets <- train |> model(ETS(ts_data))
report(fit_ets)

fit_ets |>
  gg_tsresiduals(lag_max = 20)


####### compare ARIMA vs ETS ###########

train <- ts_data |>
  filter(Datetime < as.POSIXct("2018-05-01"))

fit_arima <- train |> model(ARIMA(PJME_MW))
report(fit_arima)
fit_arima |> gg_tsresiduals(lag_max = 50)

augment(fit_arima) |>
  features(.innov, ljung_box, lag = 8, dof = 5)

# ETS
train_clean <- train |>
  fill(PJME_MW, .direction = "downup")
fit_ets <- train_clean |> model(ETS(PJME_MW))
report(fit_ets)
fit_ets |>
  gg_tsresiduals(lag_max = 16)
augment(fit_ets) |>
  features(.innov, ljung_box, lag = 8)

# Generate forecasts and compare accuracy over the test set
bind_rows(
  fit_arima |> accuracy(),
  fit_ets |> accuracy(),
  fit_arima |> forecast(h = 24) |> accuracy(ts_data),
  fit_ets |> forecast(h = 24) |> accuracy(ts_data)
) |>
  select(-ME, -MPE, -ACF1)


# graph
ts_data |>
  model(ARIMA(PJME_MW)) |>
  filter(year(Datetime) == 2018) |>
  forecast(h=24) |>
  autoplot(ts_data) +
  labs(title = "PJME with forecast 24 hrs",
       y = "MW")
