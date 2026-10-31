#!/usr/bin/env python
# coding: utf-8

# #  MOBILE PHONE PRICE PREDICTION
# 

# In[12]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# In[13]:


# Load the dataset

EDA = pd.read_csv("C:\\Users\\adity\\OneDrive\\Desktop\\NextHikes Project\\Mobile Price Prediction Project 4\\Processed_Flipdata - Processed_Flipdata.csv")


# In[14]:


# Display the first five rows

print("FIRST FIVE ROWS")
print(EDA.head())


# In[15]:


# Display the last five rows

print("\n5. Last Five Rows:")
print(EDA.tail())


# EDA.head() and EDA.tail() show the first and last five rows so you can verify that the file loaded properly

# In[16]:


# Display data types and information

print("\nDATASET INFORMATION")
EDA.info()


# EDA.info() shows column names, non-null counts, data types and memory use

# In[17]:


# Check the number of rows and columns

print("\nDataset shape (rows, columns):")
print(EDA.shape)


# EDA.shape gives the number of rows and columns; EDA.columns.tolist() prints column names.

# In[18]:


# Check Column names

print("\nColumn names:")
print(EDA.columns.tolist())


# In[19]:


# Check missing values in each column

print("\n6. Missing Values:")
print(EDA.isnull().sum())


# In[20]:


# Calculate the total number of missing values

print("\n7. Total Missing Values:")
print(EDA.isnull().sum().sum())


# EDA.isnull().sum() counts missing cells per column; .sum().sum() gives the total missing cells.

# In[21]:


# Check duplicate rows

print("\n8. Total Duplicate Rows:")
print(EDA.duplicated().sum())


# EDA.duplicated().sum() counts fully duplicated rows.

# In[22]:


# Generate statistical summaries

print("\n9. Statistical Summary:")
print(EDA.describe(include="all").T)


# EDA.describe(include='all').T gives summary statistics for numeric and categorical columns.

# In[23]:


# Count unique values in each column

print("\n10. Unique Values Per Column:")
print(EDA.nunique())


# EDA.nunique() counts distinct values per column; EDA.dtypes displays data types.

# In[24]:


# Check data types separately

print("\n11. Data Types:")
print(EDA.dtypes)


# ### Data Cleaning and Preprocessing

# In[25]:


# Create a copy so the original DataFrame remains unchanged

EDA = EDA.copy()


# EDA = EDA.copy() creates a separate working copy.

# In[26]:


# Remove unnecessary index column if it exists

if "Unnamed: 0" in EDA.columns:
    EDA = EDA.drop(columns=["Unnamed: 0"])
    print("\nUnnecessary index column removed.")


# The if statement removes 'Unnamed: 0' if present. It is usually an old row-index column, not a real mobile specification.

# In[27]:


# Clean the target price column

if "Prize" not in EDA.columns:
    raise KeyError("The target column 'Prize' was not found.")

EDA["Prize"] = (
    EDA["Prize"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

EDA["Prize"] = pd.to_numeric(EDA["Prize"], errors="coerce")


# The Prize check raises a clear error if the target column is missing.

# In[28]:


# Check missing values after price conversion

print("\nMissing values after price conversion:")
print(EDA.isnull().sum())


# In[29]:


# Remove rows where the target price is missing or invalid

rows_before_price_cleaning = len(EDA)

EDA = EDA.dropna(subset=["Prize"]).copy()

print(
    "\nRows removed due to missing or invalid prices:",
    rows_before_price_cleaning - len(EDA)
)


# dropna(subset=['Prize']) removes rows with unknown/invalid target prices. The before/after count reports how many were removed.

# In[30]:


# Check duplicate rows

duplicate_count = EDA.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

if duplicate_count > 0:
    EDA = EDA.drop_duplicates().copy()
    print("Exact duplicate rows removed:", duplicate_count)
else:
    print("No exact duplicate rows found.")


# duplicated() counts exact duplicate rows; drop_duplicates() removes them if found

# In[31]:


# Check missing values in remaining columns

print("\nMissing values in each column:")
print(EDA.isnull().sum())


# In[32]:


# Check numeric data types

print("\nData types after cleaning:")
print(EDA.dtypes)


# In[33]:


# Check the price range

print("\nMinimum price:", EDA["Prize"].min())
print("Maximum price:", EDA["Prize"].max())
print("Average price:", round(EDA["Prize"].mean(), 2))
print("Median price:", EDA["Prize"].median())


# In[34]:


# Inspect unusual mobile height values

if "Mobile Height" in EDA.columns:
    print("\nMobile Height summary:")
    print(pd.to_numeric(EDA["Mobile Height"], errors="coerce").describe())


# In[35]:


# Display the cleaned dataset

print("\nCleaned dataset shape:")
print(EDA.shape)


# The later prints recheck missing values and data types, report min/max/mean/median price, inspect Mobile Height if present, and show the cleaned shape.

# ### Exploratory Data Analysis (EDA) and visualisations.

# In[36]:


# Set a light-colour visualisation theme

sns.set_style("whitegrid")
sns.set_context("notebook")

plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["axes.facecolor"] = "#FAFCFF"
plt.rcParams["axes.edgecolor"] = "#D9E2F2"
plt.rcParams["grid.color"] = "#E5EAF3"
plt.rcParams["font.size"] = 10


# sns.set_style('whitegrid') and sns.set_context('notebook') choose a chart style and scale. plt.rcParams sets default backgrounds, grid colour, borders and font size.

# In[37]:


#  Check whether the target column exists

if "Prize" not in EDA.columns:
    raise KeyError("The 'Prize' column is missing. Complete Step 3 first.")


# In[38]:


# Price distribution histogram

plt.figure(figsize=(10, 6))

plt.hist(
    EDA["Prize"].dropna(),
    bins=25,
    color="#A8C7FA",
    edgecolor="#FFFFFF"
)

plt.title("Distribution of Mobile Phone Prices")
plt.xlabel("Price")
plt.ylabel("Number of Mobile Phones")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# Price histogram: plt.hist(..., bins=25) groups prices into 25 intervals. Bar height shows how many phones fall into each interval.
# 
# 

# In[39]:


# Box plot of mobile phone prices

plt.figure(figsize=(10, 4))

sns.boxplot(
    data=EDA,
    x="Prize",
    color="#CDBDFA"
)

plt.title("Mobile Phone Price Spread and Potential Outliers")
plt.xlabel("Price")
plt.tight_layout()
plt.show()


# Price box plot: sns.boxplot(...) displays the median, quartiles and potential outliers. Outliers should be investigated, not automatically deleted.

# In[48]:


# Top 15 Common Mobile Processors

processor_counts = EDA["Processor_"].value_counts().head(15)

plt.figure(figsize=(10, 6))

processor_counts.sort_values().plot(
    kind="barh",
    color="skyblue",
    edgecolor="black"
)

plt.title("Top 15 Common Mobile Processors", fontsize=15, fontweight="bold")
plt.xlabel("Number of Mobiles")
plt.ylabel("Processor")

plt.tight_layout()
plt.show()


# Processor distribution provides useful information about mobile hardware trends and can help identify whether processor type plays an important role in predicting mobile phone prices.
# 
# This bar chart shows the 15 most commonly used processors in the mobile dataset. It helps identify which processors are most frequently found in the available mobile phones.

# In[49]:


# Rear Camera Configuration vs Price

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=EDA,
    x="Rear Camera",
    y="Prize"
)

plt.title("Rear Camera Configuration vs Price")
plt.xlabel("Rear Camera")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Rear camera configuration is a useful feature for understanding mobile price variations. It can contribute to price prediction when combined with other features such as RAM, processor, battery capacity and internal storage.

# In[55]:


# Brand-wise Mobile Price Analysis

EDA["Brand"] = (
    EDA["Model"]
    .astype(str)
    .str.strip()
    .str.split()
    .str[0]
)

EDA["Prize"] = pd.to_numeric(EDA["Prize"], errors="coerce")

brand_group = EDA.groupby("Brand")["Prize"]

brand_analysis = pd.DataFrame({
    "Mobile_Count": brand_group.count(),
    "Average_Price": brand_group.mean(),
    "Maximum_Price": brand_group.max(),
    "Minimum_Price": brand_group.min()
})

brand_analysis = brand_analysis.sort_values(
    by="Average_Price",
    ascending=False
)

print("Top 15 Brands by Average Mobile Price")
print(brand_analysis.head(15).round(2))

plt.figure(figsize=(11, 6))

brand_analysis["Average_Price"].head(15).sort_values().plot(
    kind="barh",
    color="teal",
    edgecolor="black"
)

plt.title("Top 15 Brands by Average Mobile Price",
          fontsize=15, fontweight="bold")
plt.xlabel("Average Price")
plt.ylabel("Brand")
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


# Brand plays an important role in mobile phone pricing. This analysis helps understand brand-wise price differences and makes Brand a useful feature for building an accurate Mobile Price Prediction model.

# In[40]:


# Average Mobile Price by Internal Memory

if "Memory" in EDA.columns and "Prize" in EDA.columns:

    memory_data = EDA[["Memory", "Prize"]].copy()

    memory_data["Memory"] = (
        memory_data["Memory"].astype(str).str.strip()
    )

    memory_data["Prize"] = pd.to_numeric(
        memory_data["Prize"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    memory_data = memory_data.dropna(subset=["Prize"])
    memory_data = memory_data[
        memory_data["Memory"].str.lower() != "nan"
    ]
    memory_data = memory_data[
        memory_data["Memory"] != ""
    ]

    memory_summary = (
        memory_data.groupby("Memory")["Prize"]
        .mean()
        .sort_values()
    )

    if not memory_summary.empty:

        plt.figure(figsize=(10, 6))

        plt.barh(
            memory_summary.index,
            memory_summary.values,
            color="#F6D6AD",
            edgecolor="white"
        )

        plt.title("Average Mobile Price by Internal Memory")
        plt.xlabel("Average Price")
        plt.ylabel("Internal Memory")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()

    else:
        print("No valid memory and price data available.")

else:
    print("Error: 'Memory' or 'Prize' column not found.")


# 
# Average price by Memory: same process as RAM, but groups phones by internal storage category.

# In[41]:


# Average price by RAM

if "RAM" in EDA.columns:
    ram_data = EDA[["RAM", "Prize"]].copy()

    ram_data["RAM"] = ram_data["RAM"].astype(str).str.strip()
    ram_data["Prize"] = pd.to_numeric(
        ram_data["Prize"], errors="coerce"
    )

    ram_data = ram_data.dropna(subset=["RAM", "Prize"])

    ram_summary = (
        ram_data.groupby("RAM")["Prize"]
        .mean()
        .sort_values()
    )

    if not ram_summary.empty:
        plt.figure(figsize=(10, 6))

        plt.barh(
            ram_summary.index.astype(str),
            ram_summary.values,
            color="#A8DCC5",
            edgecolor="white"
        )

        plt.title("Average Mobile Price by RAM Category")
        plt.xlabel("Average Price")
        plt.ylabel("RAM")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid RAM and price data available.")
else:
    print("RAM column not found.")


# Average price by RAM: the code copies RAM and Prize, trims RAM text, converts price to numeric, removes invalid prices, groups by RAM, calculates mean price, sorts it and draws a horizontal bar chart.

# In[42]:


# Correlation heatmap for numeric columns

numeric_data = EDA.select_dtypes(include=["number"])
correlation = numeric_data.corr()

if correlation.shape[0] >= 2:
    plt.figure(figsize=(10, 7))

    image = plt.imshow(
        correlation,
        cmap="PuBuGn",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(image, label="Correlation Coefficient")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    


# Correlation heatmap: select_dtypes(include=['number']) selects numeric columns; .corr() calculates pairwise correlations from -1 to +1; plt.imshow() visualizes the matrix and plt.colorbar() adds the scale.

# In[67]:


# Average price by processor

if "Processor_" in EDA.columns:
    processor_price = (
        EDA.groupby("Processor_")["Prize"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    processor_price.plot(kind="bar")
    plt.title("Top 10 Processors by Average Mobile Price")
    plt.xlabel("Processor")
    plt.ylabel("Average Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[68]:


# Average price by AI Lens

if "AI Lens" in EDA.columns:
    ai_price = EDA.groupby("AI Lens")["Prize"].mean()

    plt.figure(figsize=(8, 5))
    ai_price.plot(kind="bar")
    plt.title("Average Price by AI Lens")
    plt.xlabel("AI Lens")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[69]:


# Display summary statistics
print("\nPrice Summary:")
print(EDA["Prize"].describe())


# Price summary: EDA['Prize'].describe() reports count, mean, standard deviation, minimum, quartiles and maximum.

# ###  Feature Engineering

# In[70]:



from sklearn.preprocessing import LabelEncoder


# In[71]:


# Create a copy of the EDA dataset

ML_data = EDA.copy()


# In[72]:


# Remove unnecessary index column if it exists

if "Unnamed: 0" in ML_data.columns:
    ML_data = ML_data.drop("Unnamed: 0", axis=1)


# In[73]:


# Convert text columns into numerical values

label_encoders = {}

for column in ML_data.columns:
    if ML_data[column].dtype == "object":
        encoder = LabelEncoder()

        ML_data[column] = encoder.fit_transform(
            ML_data[column].astype(str)
        )

        label_encoders[column] = encoder


# In[74]:


# Display the processed dataset

print("Feature Engineering Completed Successfully!")
print("\nDataset Shape:", ML_data.shape)


# In[75]:


print("\nProcessed Dataset:")
print(ML_data.head())


# In[76]:


print("\nData Types:")
print(ML_data.dtypes)


# LabelEncoder converts each text category into an integer code and stores each encoder in label_encoders. However, this section's ML_data is overwritten later by ML_data = EDA.copy(), so these encoded values are not used by your final model. For unordered categories such as colour or model name, arbitrary integer codes can imply a false order; one-hot encoding is usually safer for this project.

# In[77]:


# Create numerical RAM, memory and battery columns

import re

ML_data = EDA.drop(
    columns=["Prize", "Price_Category", "Unnamed: 0"],
    errors="ignore"
).copy()


def extract_number(value):
    numbers = re.findall(r"\d+\.?\d*", str(value))

    if numbers:
        return float(numbers[0])

    return np.nan


def convert_memory(value):
    value = str(value).upper()
    number = extract_number(value)

    if pd.isna(number):
        return np.nan

    if "TB" in value:
        return number * 1024

    return number


ML_data["RAM_GB"] = ML_data["RAM"].apply(extract_number)

ML_data["Memory_GB"] = (
    ML_data["Memory"].apply(convert_memory)
)

ML_data["Battery_mAh"] = (
    ML_data["Battery_"].apply(extract_number)
)

ML_data["Mobile_Height_Num"] = (
    ML_data["Mobile Height"].apply(extract_number)
)


# Numerical feature extraction converts unstructured mobile specifications into meaningful numerical features. This improves data consistency and helps machine learning models analyse the relationship between mobile specifications and their prices.

# In[78]:


# Camera Megapixel Extraction

def extract_camera_mp(value):

    numbers = re.findall(
        r"(\d+\.?\d*)\s*MP",
        str(value).upper()
    )

    if numbers:
        return sum(float(x) for x in numbers)

    return np.nan


ML_data["Rear_Camera_MP"] = (
    ML_data["Rear Camera"].apply(extract_camera_mp)
)

ML_data["Front_Camera_MP"] = (
    ML_data["Front Camera"].apply(extract_camera_mp)
)


# Camera megapixel extraction transforms text-based camera specifications into numerical features. These features help analyse how camera capabilities relate to smartphone prices and may improve the performance of the Mobile Price Prediction model.

# In[79]:


# Camera Megapixel Calculation

ML_data["Total_Camera_MP"] = (
    ML_data["Rear_Camera_MP"] +
    ML_data["Front_Camera_MP"]
)


# Total Camera MP is an engineered feature that combines front and rear camera megapixels into a single numerical variable. It can help improve feature analysis and potentially support better mobile price prediction.

# In[80]:


# RAM and Internal Memory Product

ML_data["RAM_Memory_Product"] = (
    ML_data["RAM_GB"] *
    ML_data["Memory_GB"]
)


# RAM_Memory_Product is an engineered feature that combines RAM and internal storage information. It may help improve the model's ability to identify pricing patterns and predict mobile phone prices.

# In[81]:


# Storage to RAM Ratio

ML_data["Storage_RAM_Ratio"] = (
    ML_data["Memory_GB"] /
    ML_data["RAM_GB"].replace(0, np.nan)
)


# Storage_RAM_Ratio is an engineered feature that represents the relationship between internal storage and RAM. It provides additional information about smartphone hardware configurations and may contribute to better mobile price prediction.

# In[82]:


# Battery to RAM Ratio

ML_data["Battery_RAM_Ratio"] = (
    ML_data["Battery_mAh"] /
    ML_data["RAM_GB"].replace(0, np.nan)
)


# Battery_RAM_Ratio is an engineered feature that combines battery and RAM information into a single numerical variable. It can support deeper feature analysis and potentially improve mobile price prediction.

# In[83]:


# Brand Extraction

ML_data["Brand"] = (
    ML_data["Model"].astype(str)
    .str.strip()
    .str.split()
    .str[0]
)


# Brand extraction separates manufacturer information from mobile model names. It creates a useful categorical feature that can help the Mobile Price Prediction model learn brand-wise pricing patterns.

# In[84]:


# AI Lens availability

ML_data["AI_Lens_Available"] = (
    ~ML_data["AI Lens"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["no", "false", "0", "none", "nan", ""])
).astype(int)


# AI Lens Availability is a useful engineered feature that simplifies categorical data and helps the machine learning model analyse the impact of AI-based camera features on mobile prices.

# In[85]:


# Check all created features

print("Original Dataset Shape:", EDA.shape)

print("New Feature Dataset Shape:", ML_data.shape)

print("\nNew Feature Columns:")

print(ML_data.columns.tolist())

print("\nNew Features Preview:")

ML_data.head(10)


# ### Train and Test the Machine Learning Model

# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[87]:


# Prepare the dataset

ML_data = EDA.copy()


# In[88]:


# Remove unnecessary column

if "Unnamed: 0" in ML_data.columns:
    ML_data = ML_data.drop("Unnamed: 0", axis=1)


# In[89]:


# Convert Prize into numeric values

ML_data["Prize"] = pd.to_numeric(
    ML_data["Prize"].astype(str).str.replace(",", ""),
    errors="coerce"
)


# In[90]:


# Remove rows without a valid price

ML_data = ML_data.dropna(subset=["Prize"])


# In[91]:


# Separate features and target

X = ML_data.drop("Prize", axis=1)
y = ML_data["Prize"]


# In[92]:


# Convert categorical columns into numeric columns

X = pd.get_dummies(X, dummy_na=True)


# In[93]:


# Replace infinite values and handle missing data

X = X.replace([np.inf, -np.inf], np.nan)


# In[94]:


# Remove columns that contain no usable values

X = X.dropna(axis=1, how="all")


# In[95]:


# Fill remaining missing values with each column's median

X = X.fillna(X.median())


# In[96]:


# Split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Number of features:", X_train.shape[1])


# In[97]:


# Train the model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# In[98]:


# Predict prices for test data

y_pred = model.predict(X_test)


# In[99]:


# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared Score:", round(r2, 3))


# model.predict(X_test) generates predictions for test records that were not used for fitting.
# 
# MAE (Mean Absolute Error) is the average absolute difference between actual and predicted price. Lower is better and it uses the target's price units.
# 
# RMSE (Root Mean Squared Error) is the square root of mean squared errors; it penalizes large errors more heavily. Lower is better.
# 
# R² compares the model with a baseline that predicts the test-set mean. Closer to 1 is generally better; 0 is roughly baseline-level and a negative score means worse than that baseline.

# ### Model Evaluation and Visualization

# In[100]:


# Compare actual and predicted prices

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("Actual vs Predicted Prices:")
print(results.head(10))


# This code places the real mobile prices and the model's predicted prices side by side so you can compare them and see how accurately your model predicts prices.

# In[101]:


# Plot actual vs predicted prices

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    color="teal",
    alpha=0.7
)


# This code creates a scatter plot to compare the actual mobile prices with the prices predicted by your machine learning model.

# In[102]:


# Reference line for perfect predictions
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--",
    color="red"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Mobile Prices")
plt.tight_layout()
plt.show()


# This code adds a red dashed reference line to your actual vs. predicted mobile prices graph. It helps you identify how accurately your machine learning model predicts prices.

# In[103]:


# Display the most important features
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

print("\nTop 10 Important Features:")
print(importance)


# This code identifies the 10 most important features used by your machine learning model to predict mobile prices.

# In[104]:


# Plot feature importance
plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"],
    color="cornflowerblue"
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Features Influencing Price Predictions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# This code creates a horizontal bar chart showing the 10 most important features used by your machine learning model to predict mobile prices.

# ### New Mobile Price Prediction

# In[105]:


# Enter mobile specifications

new_mobile = {}

for column in EDA.drop(columns=["Prize"], errors="ignore").columns:
    value = input("Enter " + column + ": ")
    new_mobile[column] = value


# This code allows the user to enter the specifications of a new mobile phone through keyboard input. These details can later be used by your machine learning model to predict the mobile's price.

# In[106]:


# Create a DataFrame from the entered values

new_mobile_df = pd.DataFrame([new_mobile])


# In[107]:


# Convert categorical values using the same feature format

new_mobile_encoded = pd.get_dummies(new_mobile_df, dummy_na=True)


# In[108]:


# Match the features used during model training
new_mobile_encoded = new_mobile_encoded.reindex(
    columns=X_train.columns,
    fill_value=0
)


# In[109]:


# Predict the price

predicted_price = model.predict(new_mobile_encoded)[0]

print("\n--------------------------------")
print("Estimated Mobile Price:", round(predicted_price, 2))
print("--------------------------------")


# This code takes the mobile specifications entered by the user, converts them into a format suitable for the machine learning model, predicts the price, and displays the estimated result.

# ### Save the model

# In[110]:


import pickle


# In[111]:


# Save the trained model
with open("mobile_price_model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[112]:


# Save the training feature columns
with open("mobile_price_features.pkl", "wb") as file:
    pickle.dump(list(X_train.columns), file)

print("Model saved successfully!")
print("File 1: mobile_price_model.pkl")
print("File 2: mobile_price_features.pkl")


# In[113]:


with open("mobile_price_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("mobile_price_features.pkl", "rb") as file:
    saved_features = pickle.load(file)

print("Saved model loaded successfully!")
print("Number of features:", len(saved_features))


# This code saves your trained mobile price prediction model and its feature names into files. Later, you can load these files and reuse the model without training it again.

# ### Mobile Price Prediction Interface

# In[114]:


import ipywidgets as widgets
from IPython.display import display, clear_output


# In[115]:


# Create input fields from the original dataset

feature_columns = list(X_train.columns)
original_features = EDA.drop(
    columns=["Prize", "Unnamed: 0"],
    errors="ignore"
)

input_widgets = {}

for column in original_features.columns:
    values = original_features[column].dropna().unique()

    if len(values) > 0 and len(values) <= 50:
        options = sorted([str(value) for value in values])
        input_widgets[column] = widgets.Dropdown(
            options=options,
            description=column + ":",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px")
        )
    else:
        input_widgets[column] = widgets.Text(
            value="",
            description=column + ":",
            placeholder="Enter " + column,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px")
        )

predict_button = widgets.Button(
    description="Predict Mobile Price",
    button_style="success",
    icon="check"
)

output = widgets.Output()

def predict_price(button):
    with output:
        clear_output()

        try:
            # Read the user's selected specifications
            entered_data = {}

            for column, widget in input_widgets.items():
                entered_data[column] = widget.value

            new_mobile = pd.DataFrame([entered_data])

            # Match the training preprocessing
            new_mobile_encoded = pd.get_dummies(
                new_mobile,
                dummy_na=True
            )

            new_mobile_encoded = new_mobile_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Predict price
            predicted_price = model.predict(
                new_mobile_encoded
            )[0]

            print("Estimated Mobile Price: {:.2f}".format(
                predicted_price
            ))

            print("\nNote: This is an estimate, not a guaranteed market price.")

        except Exception as error:
            print("Prediction error:", error)

predict_button.on_click(predict_price)


# In[116]:


# Display the interface

display(widgets.VBox(list(input_widgets.values())))
display(predict_button, output)


# This code creates an interactive mobile price prediction interface using Python's ipywidgets. Instead of typing specifications into separate input() prompts, users can select or enter mobile details in a form and click a button to predict the estimated price.

# ### Model performance report

# In[117]:


# Generate predictions using the test data
y_pred = model.predict(X_test)


# In[118]:


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("===== MOBILE PRICE PREDICTION REPORT =====")
print("Total test samples:", len(y_test))
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R-squared Score:", round(r2, 3))


# MAE tells you the average absolute error, RMSE gives more weight to large errors, and R² indicates how well the model explains price variation. Together, these metrics provide a useful overview of model performance.

# In[119]:


# Create a comparison table

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

comparison["Absolute Error"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
).abs()

print("\nSample Predictions:")
print(comparison.head(10).round(2))


# This code creates a table that compares the actual mobile prices with the predicted prices and calculates the error for each prediction. It helps you understand how accurately your machine learning model predicts mobile prices.

# In[120]:


# Plot prediction errors
plt.figure(figsize=(8, 5))
plt.hist(comparison["Absolute Error"], bins=15, edgecolor="black")
plt.xlabel("Absolute Prediction Error")
plt.ylabel("Number of Phones")
plt.title("Distribution of Mobile Price Prediction Errors")
plt.tight_layout()
plt.show()


# This code creates a histogram to show the distribution of mobile price prediction errors. It helps you understand how large or small the errors are across your model's predictions.

# In[121]:


# Save the trained model
with open("mobile_price_model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[122]:


# Save the feature names used during training
with open("mobile_price_features.pkl", "wb") as file:
    pickle.dump(list(X_train.columns), file)


# In[123]:


# Generate final test predictions
y_pred = model.predict(X_test)


# In[124]:


# Save actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Absolute Error"] = (
    results["Actual Price"] - results["Predicted Price"]
).abs()

results.to_csv("mobile_price_predictions.csv", index=False)


# In[125]:


# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


# In[126]:


# Save the project summary
with open("project_summary.txt", "w") as file:
    file.write("MOBILE PRICE PREDICTION PROJECT\n")
    file.write("==============================\n")
    file.write("Algorithm: Random Forest Regressor\n")
    file.write("Training samples: {}\n".format(len(X_train)))
    file.write("Testing samples: {}\n".format(len(X_test)))
    file.write("MAE: {:.2f}\n".format(mae))
    file.write("RMSE: {:.2f}\n".format(rmse))
    file.write("R-squared: {:.3f}\n".format(r2))

print("Project files saved successfully!")
print("\n1. mobile_price_model.pkl")
print("2. mobile_price_features.pkl")
print("3. mobile_price_predictions.csv")
print("4. project_summary.txt")


# This code creates a text file named project_summary.txt and saves important information about your Mobile Price Prediction project, including the algorithm used, dataset sizes, and model performance metrics.

# In[ ]:




