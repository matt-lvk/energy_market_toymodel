# Load required packages
library(fpp3)
library(readr)
library(tsibble)
library(dplyr)
library(forecast)

### From Hyndman ######







##########################

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



ts_data_daily <- ts_data |>
  filter(year(Datetime) == 2014) |>
  index_by(Date = date(Datetime)) |>
  summarise(
    Demand = sum(PJME_MW) / 1e3,
    Temperature = max(Temperature),
    Holiday = any(Holiday)
  ) |>
  mutate(Day_Type = case_when(
    Holiday ~ "Holiday",
    wday(Date) %in% 2:6 ~ "Weekday",
    TRUE ~ "Weekend"
  ))

vic_elec_daily |>
  ggplot(aes(x = Temperature, y = Demand, colour = Day_Type)) +
  geom_point() +
  labs(y = "Electricity demand (GW)",
       x = "Maximum daily temperature")