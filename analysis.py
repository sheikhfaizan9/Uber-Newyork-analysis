# If not installed, run: pip install pandas numpy matplotlib seaborn

# ----------------------------------------------------------
# Importing the libraries we need
# ----------------------------------------------------------
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For better visualizations

# ----------------------------------------------------------
# Load the dataset from a CSV file
# ----------------------------------------------------------
dataset = pd.read_csv(r'C:\Users\sheik\UberProjectpython\uber_nyc_enriched.csv')

# Show the first few rows of the dataset to see what it looks like
print("Here are the first few rows of the dataset:")
print(dataset.head())

# Check how many rows and columns we have, and get some info about the dataset
print("\nShape of the dataset (rows, columns):", dataset.shape)
print("\nDataset Info:")
print(dataset.info())

# ----------------------------------------------------------
# Preprocessing and Feature Engineering
# ----------------------------------------------------------
# Convert the 'pickup_dt' column to datetime format
if 'pickup_dt' in dataset.columns:
    dataset['pickup_dt'] = pd.to_datetime(dataset['pickup_dt'], errors='coerce')

# Create new features from the datetime
dataset['date'] = pd.DatetimeIndex(dataset['pickup_dt']).date  # Extract just the date
dataset['time'] = pd.DatetimeIndex(dataset['pickup_dt']).hour  # Extract the hour

# Categorize the time into parts of the day
dataset['day-night'] = pd.cut(x=dataset['time'], bins=[0, 10, 15, 19, 24], labels=['Morning', 'Afternoon', 'Evening', 'Night'])

# Remove any missing or duplicate data
dataset.dropna(inplace=True)  # Remove rows with missing values
dataset.drop_duplicates(inplace=True)  # Remove duplicate rows

# Add columns for the day of the week and month
dataset['DAY'] = dataset['pickup_dt'].dt.weekday  # 0=Mon, 6=Sun
dataset['MONTH'] = dataset['pickup_dt'].dt.month  # 1=Jan, 12=Dec

# Map numbers to day and month names
day_label = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dataset['DAY'] = dataset['DAY'].map(day_label)
dataset['MONTH'] = dataset['MONTH'].map(month_label)

# ----------------------------------------------------------
# 1. Monthly Trend of Rides (Line Plot)
# ----------------------------------------------------------
monthly_rides = dataset.groupby('MONTH')['pickups'].sum().reindex([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])

plt.figure(figsize=(12, 5))
monthly_rides.plot(marker='o')
plt.title('Monthly Ride Trend')
plt.xlabel('Month')
plt.ylabel('Total Pickups')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2. Pie Chart: Ride Distribution by Time of Day
# ----------------------------------------------------------
plt.figure(figsize=(6, 6))
dataset['day-night'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    shadow=True
)
plt.title('Ride Distribution: Morning vs Afternoon vs Evening vs Night')
plt.ylabel('')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3. Weather vs Ride Analysis (Rainfall impact)
# ----------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.scatterplot(x='pcp01', y='pickups', data=dataset, alpha=0.5, label='Rainfall 1hr')
sns.scatterplot(x='pcp06', y='pickups', data=dataset, alpha=0.5, label='Rainfall 6hr')
sns.scatterplot(x='pcp24', y='pickups', data=dataset, alpha=0.5, label='Rainfall 24hr')
plt.title('Impact of Rainfall on Number of Pickups')
plt.xlabel('Rainfall (inches)')
plt.ylabel('Number of Pickups')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. Optional: Bar Plot for Rides by Day of Week
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(x='DAY', data=dataset, order=['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.title('Number of Rides by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Rides')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. Prepare CSV for Tableau Dashboard
# ----------------------------------------------------------
dashboard_data = dataset[[
    'pickup_dt', 'borough', 'pickups', 'temp', 'pcp01', 'pcp06', 'pcp24', 'DAY', 'MONTH', 'day-night'
]]

# Save the dashboard-ready data to a CSV file
dashboard_data.to_csv(r'C:\Users\sheik\UberProjectpython\Tableau_Uber_DashboardData.csv', index=False)
print("✅ Data for Tableau saved as 'Tableau_Uber_DashboardData.csv'")
