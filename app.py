# ==========================================================
# Sales Forecasting Dashboard
# Streamlit Application
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


# Streamlit Configuration


st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence System",
    page_icon="📈",
    layout="wide"
)

# Title


st.title("📈 Sales Forecasting & Demand Intelligence System")
st.markdown("### End-to-End Retail Sales Forecasting Dashboard")


# Load Dataset

@st.cache_data
def load_data():

    df = pd.read_csv("Superstores_dataset.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month Number"] = df["Order Date"].dt.month

    return df


df = load_data()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segments"

    ]

)


# PAGE 1
if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    # Filters
    col1, col2 = st.columns(2)

    with col1:

        region = st.selectbox(

            "Select Region",

            ["All"] + sorted(df["Region"].unique().tolist())

        )

    with col2:

        category = st.selectbox(

            "Select Category",

            ["All"] + sorted(df["Category"].unique().tolist())

        )

    filtered = df.copy()

    if region != "All":

        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":

        filtered = filtered[
            filtered["Category"] == category
        ]

    
    # KPI Cards
    
    total_sales = filtered["Sales"].sum()

    total_orders = filtered["Order ID"].nunique()

    total_categories = filtered["Category"].nunique()

    total_regions = filtered["Region"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Total Sales",

        f"${total_sales:,.2f}"

    )

    c2.metric(

        "Orders",

        total_orders

    )

    c3.metric(

        "Categories",

        total_categories

    )

    c4.metric(

        "Regions",

        total_regions

    )

    st.markdown("---")

  
    # Sales By Year
    st.subheader("📅 Total Sales by Year")

    yearly = (

        filtered.groupby("Year")["Sales"]

        .sum()

        .reset_index()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(

        data=yearly,

        x="Year",

        y="Sales",

        palette="viridis",

        ax=ax

    )

    ax.set_xlabel("Year")

    ax.set_ylabel("Sales")

    st.pyplot(fig)


    # Monthly Trend
    st.subheader("📈 Monthly Sales Trend")

    monthly = (

        filtered.groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(

        monthly["Order Date"],

        monthly["Sales"],

        linewidth=2

    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Sales")

    ax.grid(True)

    st.pyplot(fig)

    
    # Region Sales
    st.subheader("🌍 Sales by Region")

    region_sales = (

        filtered.groupby("Region")["Sales"]

        .sum()

        .reset_index()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(

        data=region_sales,

        x="Region",

        y="Sales",

        palette="Set2",

        ax=ax

    )

    st.pyplot(fig)


    # Category Sales
    st.subheader("📦 Sales by Category")

    category_sales = (

        filtered.groupby("Category")["Sales"]

        .sum()

        .reset_index()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(

        data=category_sales,

        x="Category",

        y="Sales",

        palette="coolwarm",

        ax=ax

    )

    st.pyplot(fig)


# PAGE 2

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.markdown(
        "Forecast future sales using the **best-performing XGBoost model**."
    )

   
    # Forecast Type

    forecast_type = st.selectbox(

        "Forecast By",

        [

            "Category",

            "Region"

        ]

    )

    
    # Select Value

    if forecast_type == "Category":

        selected = st.selectbox(

            "Select Category",

            sorted(df["Category"].unique())

        )

        forecast_df = df[
            df["Category"] == selected
        ]

    else:

        selected = st.selectbox(

            "Select Region",

            sorted(df["Region"].unique())

        )

        forecast_df = df[
            df["Region"] == selected
        ]

    # Forecast Horizon

    months = st.slider(

        "Forecast Horizon (Months)",

        min_value=1,

        max_value=3,

        value=3

    )

    
    # Prepare Monthly Sales
    
    monthly = (

        forecast_df

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    monthly.rename(

        columns={

            "Order Date":"Date"

        },

        inplace=True

    )

    monthly["Month"] = monthly["Date"].dt.month
    monthly["Quarter"] = monthly["Date"].dt.quarter

  
    # Season

    def get_season(month):

        if month in [12,1,2]:
            return 1

        elif month in [3,4,5]:
            return 2

        elif month in [6,7,8,9]:
            return 3

        else:
            return 4

    monthly["Season"] = monthly["Month"].apply(get_season)

    
    # Lag Features
    
    monthly["Lag1"] = monthly["Sales"].shift(1)

    monthly["Lag2"] = monthly["Sales"].shift(2)

    monthly["Lag3"] = monthly["Sales"].shift(3)

    monthly["RollingMean3"] = (

        monthly["Sales"]

        .rolling(3)

        .mean()

    )

    monthly.dropna(inplace=True)
    if len(monthly) < 8:
        st.warning("Not enough historical data available for forecasting.")
        st.stop()
    
    # Features
    
    features = [

        "Lag1",

        "Lag2",

        "Lag3",

        "RollingMean3",

        "Month",

        "Quarter",

        "Season"

    ]

    X = monthly[features]

    y = monthly["Sales"]

    X_train = X.iloc[:-3]

    X_test = X.iloc[-3:]

    y_train = y.iloc[:-3]

    y_test = y.iloc[-3:]

    
    # Train Model
    
    model = XGBRegressor(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=3,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    
    # Test Prediction
    
    pred = model.predict(X_test)

    mae = mean_absolute_error(

        y_test,

        pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            pred

        )

    )

    
    # Future Forecast

    history = monthly.copy()

    future_predictions = []

    future_dates = pd.date_range(

        monthly["Date"].iloc[-1]

        +

        pd.offsets.MonthEnd(1),

        periods=months,

        freq="ME"

    )

    for future in future_dates:

        lag1 = history["Sales"].iloc[-1]

        lag2 = history["Sales"].iloc[-2]

        lag3 = history["Sales"].iloc[-3]

        rolling = history["Sales"].tail(3).mean()

        future_X = pd.DataFrame({

            "Lag1":[lag1],

            "Lag2":[lag2],

            "Lag3":[lag3],

            "RollingMean3":[rolling],

            "Month":[future.month],

            "Quarter":[future.quarter],

            "Season":[get_season(future.month)]

        })

        value = model.predict(future_X)[0]

        future_predictions.append(value)

        history.loc[len(history)] = [

            future,

            value,

            future.month,

            future.quarter,

            get_season(future.month),

            lag1,

            lag2,

            lag3,

            rolling

        ]

    
    # Forecast Table

    forecast_table = pd.DataFrame({

        "Forecast Month":future_dates,

        "Predicted Sales":future_predictions

    })

    st.subheader("Forecast Results")

    st.dataframe(forecast_table)

   
    # Forecast Chart
    
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(

        monthly["Date"],

        monthly["Sales"],

        label="Historical",

        linewidth=2

    )

    ax.plot(

        future_dates,

        future_predictions,

        marker="o",

        linewidth=3,

        label="Forecast"

    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

  
    # Metrics

    c1, c2 = st.columns(2)

    c1.metric(

        "MAE",

        f"{mae:,.2f}"

    )

    c2.metric(

        "RMSE",

        f"{rmse:,.2f}"

    )

    st.success(

        "XGBoost was selected because it achieved the lowest forecasting errors among all evaluated models."

    )

elif page == "Anomaly Report":

    st.header("🚨 Anomaly Report")

    st.write(
        "Weekly sales anomalies detected using Isolation Forest and Rolling Z-Score."
    )

    weekly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Isolation"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    weekly_sales["Anomaly"] = (
        weekly_sales["Isolation"] == -1
    )

    weekly_sales["RollingMean"] = (
        weekly_sales["Sales"]
        .rolling(8)
        .mean()
    )

    weekly_sales["RollingStd"] = (
        weekly_sales["Sales"]
        .rolling(8)
        .std()
    )

    weekly_sales["RollingZ"] = (
        weekly_sales["Sales"] -
        weekly_sales["RollingMean"]
    ) / weekly_sales["RollingStd"]

    weekly_sales["Z_Anomaly"] = (
        weekly_sales["RollingZ"].abs() > 2
    )

    st.subheader("Isolation Forest Anomalies")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales",
        linewidth=2
    )

    ax.scatter(
        weekly_sales.loc[
            weekly_sales["Anomaly"],
            "Order Date"
        ],
        weekly_sales.loc[
            weekly_sales["Anomaly"],
            "Sales"
        ],
        color="red",
        s=100,
        label="Anomaly"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Rolling Z-Score Anomalies")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        linewidth=2,
        label="Weekly Sales"
    )

    ax.scatter(
        weekly_sales.loc[
            weekly_sales["Z_Anomaly"],
            "Order Date"
        ],
        weekly_sales.loc[
            weekly_sales["Z_Anomaly"],
            "Sales"
        ],
        color="green",
        marker="D",
        s=90,
        label="Z-Score Anomaly"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Detected Anomalies (Isolation Forest)")

    anomaly_table = weekly_sales.loc[
        weekly_sales["Anomaly"],
        ["Order Date", "Sales"]
    ].copy()

    anomaly_table.rename(
        columns={
            "Order Date": "Anomaly Date",
            "Sales": "Weekly Sales"
        },
        inplace=True
    )

    anomaly_table["Weekly Sales"] = (
        anomaly_table["Weekly Sales"]
        .round(2)
    )

    st.dataframe(
        anomaly_table,
        use_container_width=True
    )

    csv = anomaly_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Anomaly Report",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

    st.info(
        "Isolation Forest identifies both unusually high and unusually low sales weeks, while the Rolling Z-Score highlights weeks with statistically significant deviations from the recent trend."
    )

elif page == "Demand Segments":

    st.header("📦 Product Demand Segments")

    st.write(
        "Product sub-categories are grouped using K-Means clustering based on sales characteristics."
    )

    product_features = (
        df.groupby("Sub-Category")
        .agg(
            Total_Sales=("Sales", "sum"),
            Average_Order_Value=("Sales", "mean")
        )
        .reset_index()
    )

    monthly_subcat = (
        df.groupby(
            [
                "Sub-Category",
                pd.Grouper(key="Order Date", freq="ME")
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    volatility = (
        monthly_subcat
        .groupby("Sub-Category")["Sales"]
        .std()
        .reset_index()
        .rename(columns={"Sales": "Sales_Volatility"})
    )

    year_sales = (
        df.groupby(
            [
                "Sub-Category",
                "Year"
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    growth = []

    for sub in year_sales["Sub-Category"].unique():

        temp = (
            year_sales[
                year_sales["Sub-Category"] == sub
            ]
            .sort_values("Year")
        )

        first = temp.iloc[0]["Sales"]
        last = temp.iloc[-1]["Sales"]

        if first == 0:
            growth_rate = 0
        else:
            growth_rate = ((last - first) / first) * 100

        growth.append([sub, growth_rate])


    growth = pd.DataFrame(
        growth,
        columns=[
            "Sub-Category",
            "Growth_Rate"
        ]
    )

    cluster_df = product_features.merge(
        volatility,
        on="Sub-Category"
    )

    cluster_df = cluster_df.merge(
        growth,
        on="Sub-Category"
    )

    features = [
        "Total_Sales",
        "Growth_Rate",
        "Sales_Volatility",
        "Average_Order_Value"
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        cluster_df[features]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster_df["Cluster"] = kmeans.fit_predict(
        scaled
    )

    labels = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand",
        2: "Low Volume, High Volatility",
        3: "Declining Demand"
    }

    cluster_df["Demand Cluster"] = (
        cluster_df["Cluster"]
        .map(labels)
    )

    pca = PCA(n_components=2)

    components = pca.fit_transform(
        scaled
    )

    cluster_df["PC1"] = components[:, 0]
    cluster_df["PC2"] = components[:, 1]

    st.subheader("Demand Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=cluster_df,
        x="PC1",
        y="PC2",
        hue="Demand Cluster",
        s=180,
        ax=ax
    )

    for _, row in cluster_df.iterrows():

        ax.text(
            row["PC1"] + 0.05,
            row["PC2"] + 0.05,
            row["Sub-Category"],
            fontsize=8
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)

    st.subheader("Sub-Category Demand Segments")

    st.dataframe(
        cluster_df[
            [
                "Sub-Category",
                "Demand Cluster",
                "Total_Sales",
                "Growth_Rate",
                "Sales_Volatility",
                "Average_Order_Value"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Recommended Stocking Strategy")

    st.markdown("""
| Demand Cluster | Recommended Strategy |
|----------------|----------------------|
| **High Volume, Stable Demand** | Maintain high inventory and prioritize replenishment. |
| **Growing Demand** | Increase inventory gradually and maintain safety stock. |
| **Low Volume, High Volatility** | Keep conservative inventory and replenish based on demand. |
| **Declining Demand** | Reduce inventory and avoid excessive stock. |
""")

    csv = cluster_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Demand Segments",
        data=csv,
        file_name="demand_segments.csv",
        mime="text/csv"
    )