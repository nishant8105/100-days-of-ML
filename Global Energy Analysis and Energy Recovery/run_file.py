from data_loading_and_eda import run_eda, run_correlation_analysis
from feature_engineering import preprocess_data
from linear_regression_model import run_linear_regression
from logistic_regression_model_and_summary import run_logistic_regression, summarize_findings

data = run_eda()
run_correlation_analysis(data)
data = preprocess_data(data)
run_linear_regression(data)
run_logistic_regression(data)
summarize_findings()
