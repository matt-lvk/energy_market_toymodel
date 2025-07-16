```mermaid
classDiagram
    %% Abstract Base Classes
    class ModelWrapper {
        <<abstract>>
        +ts: TSDataFrame
        +split_date: datetime
        +start_date_slice: datetime
        +get_rmse_mae() tuple
        +plot_forecast() void
    }

    %% Concrete Model Classes
    class ARIMAWrapper {
        +model: ARIMAResults
        +run_ARIMA(p: int, d: int, q: int) void
        +get_forecast() void
        +plot_acf_pacf() void
    }

    class ProphetWrapper {
        +prophet_model: Prophet
        +forecasted_df: DataFrame
        +run_prophet() void
        +get_forecast() DataFrame
        +plot_forecast() void
    }

    class XGBWrapper {
        +model: xgb.XGBRegressor
        +predicted_XGBoost: DataFrame
        +run_xgb() void
        +add_lagged_MA_price(hours: int, days: int) void
        +plot_importance() void
        +plot_forecast() void
    }

    %% Battery Environment Classes
    class BatteryEnv {
        +max_capacity: float
        +max_charge_discharge: float
        +initial_charge: float
        +action_space() list
        +state_space() tuple
        +reset() void
        +step(action: int) tuple
    }

    class QLearningAgent {
        +q_table: dict
        +learning_rate: float
        +discount_factor: float
        +epsilon: float
        +choose_action(state: tuple) int
        +update_q_table(state: tuple, action: int, reward: float, next_state: tuple) void
        +train(episodes: int) void
    }

    %% Backtesting Classes
    class BacktestParent {
        <<abstract>>
        +split_dates: list
        +run_backtest() void
        +calculate_metrics() dict
    }

    class BacktestXGB {
        +short_xgb_obj: XGBWrapper
        +window_xgb_forecast: list
        +plot_forecast() void
    }

    %% Utility Classes
    class Plotters {
        <<utility>>
        +plot_actual_predict(df: DataFrame, y_var: str, pred_var: str, title: str) void
        +plot_violin_ts(df: DataFrame, x_var: str, y_var: str, title: str) void
        +plot_compare_two_col(df: DataFrame, y1_var: str, y2_var: str, title: str) void
        +plotly_actual_predict(df: DataFrame, y_var: str, pred_var: str, title: str) void
    }

    class DataUtils {
        <<utility>>
        +load_price_data() DataFrame
        +clean_data() DataFrame
        +add_features() DataFrame
    }

    %% Optimization Functions
    class OptimizationModule {
        <<module>>
        +opt_battery_problem(predicted_array: array) tuple
        +solve_optimization() Problem
    }

    %% Data Files
    class DataFiles {
        <<data>>
        +Day-ahead_Prices_60min.csv
        +Actual_consumption_202112310000_202207010000_Hour.csv
        +Forecasted_generation_Day-Ahead_202112310000_202207010000_Hour.csv
    }

    %% Relationships
    ModelWrapper <|-- ARIMAWrapper : inherits
    ModelWrapper <|-- ProphetWrapper : inherits
    ModelWrapper <|-- XGBWrapper : inherits
    
    BacktestParent <|-- BacktestXGB : inherits
    
    BatteryEnv --> QLearningAgent : uses
    
    XGBWrapper --> Plotters : uses
    ARIMAWrapper --> Plotters : uses
    ProphetWrapper --> Plotters : uses
    
    BacktestXGB --> XGBWrapper : contains
    
    XGBWrapper --> DataUtils : uses
    ARIMAWrapper --> DataUtils : uses
    ProphetWrapper --> DataUtils : uses
    
    OptimizationModule --> BatteryEnv : optimizes
    
    DataFiles --> DataUtils : provides data
    
    %% Main workflow connections
    XGBWrapper --> OptimizationModule : forecasted prices
    OptimizationModule --> BatteryEnv : optimization results
    BatteryEnv --> QLearningAgent : environment