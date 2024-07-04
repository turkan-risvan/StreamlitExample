

















import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Data Analysis and Machine Learning with Streamlit")

# Load the dataset once
df = pd.read_csv("C:/Users/ACER/Desktop/ReddingRootsCaseStudy22.csv")
df_copy1 = df.copy()

# Function to grab column names based on types
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

# Initialize column names
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Sidebar for navigation
st.sidebar.title("Steps")

# Defining CSS for sidebar
sidebar_style = """
    <style>
        .css-umg8b8.e1gtdkeo0 {
            background-color: #f0f8ff; /* Mavi tonu */
        }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)
 
steps = ["1. Importing Libraries and Setting Options", 
         "2. Loading the Dataset", 
         "3. Dropping Unnecessary Columns", 
         "4. Filtering Data and Basic Information", 
         "5. Analyzing Unique Values in a Column", 
         "6. Checking Missing Values", 
         "7. Grabbing Column Names Based on Types", 
         "8. Summary Functions for Categorical and Numerical Columns", 
         "9. Outlier Detection and Handling", 
         "10. Handling Missing Values", 
         "11. Correcting Inconsistent Values", 
         "12. Encoding Categorical Data", 
         "13. Correlation Analysis", 
         "14. Predictive Modeling"]

selected_step = st.sidebar.radio("Select Step", steps)

# Perform actions based on selected step
if selected_step == "1. Importing Libraries and Setting Options":
    st.header("1. Importing Libraries and Setting Options")
    st.code("""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)
    """)

elif selected_step == "2. Loading the Dataset":
    st.header("2. Loading the Dataset")
    st.write("Dataset Head:")
    st.write(df.head())
    st.write("Dataset Shape:", df.shape)
    st.write("Dataset Info:")
    st.text(df.info())

elif selected_step == "3. Dropping Unnecessary Columns":
    st.header("3. Dropping Unnecessary Columns")
    df = df.drop(columns=["Comments", "Notes", "Unnamed: 42"])
    df = df.drop(columns=["Species", "Key"])
    st.write("Dataset after Dropping Columns:")
    st.write(df.head())

elif selected_step == "4. Filtering Data and Basic Information":
    st.header("4. Filtering Data and Basic Information")
    df = df[0:93]
    st.write("Filtered Dataset:")
    st.write(df.tail())
    st.write("Filtered Dataset Shape:", df.shape)
    st.write("Filtered Dataset Info:")
    st.text(df.info())

elif selected_step == "5. Analyzing Unique Values in a Column":
    st.header("5. Analyzing Unique Values in a Column")
    st.write("Unique Values in 'VegType':", df["VegType"].nunique())
    st.write(df["VegType"].unique())

elif selected_step == "6. Checking Missing Values":
    st.header("6. Checking Missing Values")
    missing_values = {col: df[col].isnull().sum() for col in df.columns}
    st.write("Missing Values in Each Column:")
    st.write(missing_values)

elif selected_step == "7. Grabbing Column Names Based on Types":
    st.header("7. Grabbing Column Names Based on Types")
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    st.write(f"observations: {df.shape[0]}")
    st.write(f"variables: {df.shape[1]}")
    st.write(f"cat_cols: {len(cat_cols)}")
    st.write(f"num_cols: {len(num_cols)}")
    st.write(f"cat_but_car: {len(cat_but_car)}", f"cat_but_car name: {cat_but_car}")

elif selected_step == "8. Summary Functions for Categorical and Numerical Columns":
    st.header("8. Summary Functions for Categorical and Numerical Columns")

    # Categorical Summary
    def cat_summary(dataframe, col_name):
        st.write(pd.DataFrame({col_name: dataframe[col_name].value_counts(), "ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        st.write("##############done")

    st.subheader("Categorical Summary")
    for col in cat_cols:
        cat_summary(df, col)

    # Numerical Summary
    def num_summary(dataframe, col_name, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
        st.write(dataframe[col_name].describe(quantiles).T)
        if plot:
            dataframe[col_name].hist(bins=20)
            plt.xlabel(col_name)
            plt.title(col_name)
            st.pyplot(plt)

    st.subheader("Numerical Summary")
    for col in num_cols:
        num_summary(df, col, plot=True)

elif selected_step == "9. Outlier Detection and Handling":
    st.header("9. Outlier Detection and Handling")

    # Outlier Thresholds
    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        IQR_range = quartile3 - quartile1
        up_lim = quartile3 + 1.5 * IQR_range
        low_lim = quartile1 - 1.5 * IQR_range
        return low_lim, up_lim

    for col in num_cols:
        low, up = outlier_thresholds(df, col)
        st.write(f"lower limit of {col}", f"is {low}")
        st.write(f"upper limit of {col}", f"is {up}")
        st.write("######")

    # Grabbing Outliers
    def grab_outliers(dataframe, col_name, index=False):
        low, up = outlier_thresholds(dataframe, col_name)
        count = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)][col_name].shape[0]
        st.write(col_name.upper(), f" has {count} outliers.")
        if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] != 0:
            if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
                st.write(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head(5))
            else:
                st.write(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
        if index:
            outlier_index = (dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]).index
            return outlier_index

    for col in num_cols:
        grab_outliers(df, col, index=True)

    # Checking Outliers
    def check_outlier(dataframe, col_name):
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
            return True
        else:
            return False

    for col in num_cols:
        st.write(col, check_outlier(df, col))

    # Replacing Outliers with Thresholds
    def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.90):
        low, up = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.90)
        dataframe.loc[(dataframe[variable] < low), variable] = low
        dataframe.loc[(dataframe[variable] > up), variable] = up

    for col in num_cols:
        st.write(col, check_outlier(df, col))
        if check_outlier(df, col):
            replace_with_thresholds(df, col)

    for col in num_cols:
        st.write(col, check_outlier(df, col))

elif selected_step == "10. Handling Missing Values":
    st.header("10. Handling Missing Values")
    st.write(df.isnull().sum())
    msno.matrix(df)
    plt.show()
    st.pyplot(plt)

    df.dropna(inplace=True)
    st.write(df.isnull().sum())

elif selected_step == "11. Correcting Inconsistent Values":
    st.header("11. Correcting Inconsistent Values")
    corrections = {
        "Sweetgrass\n(Muhlenbergia filipes)": "Sweetgrass",
        "Micanthus grass\n(Miscanthus sinensis)": "Micanthus grass",
        "Sea oats": "Sea oats",
        "\t-sea oats": "Sea oats",
        "-sea oats": "Sea oats",
        "no": "no",
    }

    df["VegType"] = df["VegType"].replace(corrections)
    st.write("Unique Values in 'VegType':", df["VegType"].unique())
    st.write("Number of Unique Values in 'VegType':", df["VegType"].nunique())

    df["PlantRoot"].fillna("no", inplace=True)
    st.write("Unique Values in 'PlantRoot':", df["PlantRoot"].unique())
    st.write("Missing Values in Dataset:")
    st.write(df.isnull().sum())
    st.write("Value Counts in 'VegType':")
    st.write(df["VegType"].value_counts())
    st.write("Value Counts in 'PlantRoot':")
    st.write(df["PlantRoot"].value_counts())

elif selected_step == "12. Encoding Categorical Data":
    st.header("12. Encoding Categorical Data")
    labelencoder = LabelEncoder()
    df["VegType"] = labelencoder.fit_transform(df["VegType"])
    df["PlantRoot"] = labelencoder.fit_transform(df["PlantRoot"])

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols.append("VegType")

elif selected_step == "13. Correlation Analysis":
    st.header("13. Correlation Analysis")
    corr = df[num_cols].corr()
    sns.set(rc={'figure.figsize': (20, 15)})
    sns.heatmap(corr, annot=True, cmap="RdBu")
    st.pyplot(plt)

elif selected_step == "14. Predictive Modeling":
    st.header("14. Predictive Modeling")
    X = df.drop(columns=["VegType", "PlantRoot"])
    y = df["VegType"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Add some images related to caretta caretta and plants
image_caretta_url = "https://www.cumhuriyet.com.tr/Archive/2021/8/5/1858147/kapak_121727.jpg"  # Replace with your actual URL
image_plants_url = "https://cdn.myikas.com/images/17775d73-fbcd-4aad-9210-becaabc24898/41d0b0f4-b93a-40ba-bb6c-e3788d65c5db/image_1080.webp"  # Replace with your actual URL

st.image(image_caretta_url, caption="Image of caretta caretta", use_column_width=True)
st.image(image_plants_url, caption="Image of plants", use_column_width=True)
