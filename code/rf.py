from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import timeit

def train_and_evaluate():
    data = pd.read_csv("preprocessed_screentime_analysis.csv")
    # split data into features and target variable
    X = data.drop(columns=['Usage (minutes)', 'Date'])
    y = data['Usage (minutes)']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

import statistics

num_runs = 5
execution_times = timeit.repeat(train_and_evaluate, number=1, repeat=num_runs)
avg_time = statistics.mean(execution_times)
std_dev = statistics.stdev(execution_times)

print(f"Avg execution Time: {avg_time:.2f} seconds")
print(f"Std Deviation : {std_dev:.2f} seconds")
