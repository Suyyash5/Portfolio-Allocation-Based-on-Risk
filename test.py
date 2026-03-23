import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 1. List of your file paths
files = [
   "data/ADANI_ENTERPRISES.csv",
    "data/ADANI_PORTS.csv",
    "data/APOLLO HOSPITALS.csv",
    "data/ASIAN PAINTS.csv",
    "data/AXIS_BANK.csv",
    "data/BAJAJ AUTO.csv",
    "data/BAJAJ_FINANCE.csv",
    "data/BAJAJ_FINSERV.csv",
    "data/BHARAT PETROLEUM.csv",
    "data/BHARTI_AIRTEL.csv",
    "data/BRITANNIA.csv",
    "data/CIPLA.csv",
    "data/COAL INDIA.csv",
    "data/DIVIS LAB.csv",
    "data/EICHER MOTOTRS.csv",
    "data/GRASIM.csv",
    "data/HCL_TECHNOLOIES.csv",
    "data/HDFC_BANK.csv",
    "data/HDFC_LIFE.csv",
    "data/HERO MOTOCORP.csv",
    "data/HINDALCO.csv",
    "data/HINDUSTAN UNILEVER.csv",
    "data/ICICI_BANK.csv",
    "data/INDUS INDUSTRIES.csv",
    "data/INFOSYS.csv",
    "data/ITC.csv",
    "data/JSW STEEL.csv",
    "data/KOTAK_MAHINDRA.csv",
    "data/MARUTI SUZUKI.csv",
    "data/NESTLE.csv",
    "data/NTPC.csv",
    "data/ONGC.csv",
    "data/POWERGRID.csv",
    "data/RELIANCE.csv",
    "data/SBI_BANK.csv",
    "data/SBI_LIFE.csv",
    "data/SUN_PHARMA.csv",
    "data/TATA CONSULTANCY SERVICES.csv",
    "data/TATA CONSUMER PRODUCTS.csv",
    "data/TATA MOTORS.csv",
    "data/TATA STEEL.csv",
    "data/TECH_MAHINDRA.csv",
    "data/TITAN.csv",
    "data/ULTRATECH CEMENT.csv",
    "data/UPL.csv",
    "data/WIPRO.csv",
]

results = []

for file in files:
    # Get stock name from filename
    stock_name = os.path.basename(file).replace(".csv", "")
    
    # Load and Filter
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    f_df = df[df["Date"] > "2011-01-01"].copy()
    # --- Vectorized Calculations (Faster than loops) ---
    # .pct_change() calculates: (current - previous) / previous
    # Add fill_method=None inside the parentheses
    f_df["Ret"] = f_df["Adj Close"].pct_change(fill_method=None) * 100
    
    avg_R = f_df["Ret"].mean()
    risk = f_df["Ret"].std() # Pandas .std() uses N-1 (Bessel's correction)
    
    # Append results to our list
    results.append({
        "Stock": stock_name,
        "Avg_Return": avg_R,
        "Risk_StDev": risk
    })

# 2. Create the final summary DataFrame
comp_df = pd.DataFrame(results)

# features
X = comp_df[["Avg_Return","Risk_StDev"]]

# scale data (VERY IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
comp_df['Cluster'] = kmeans.fit_predict(X_scaled)
  
# 1. Calculate Sharpe Ratio on the WHOLE dataframe first
comp_df['Sharpe'] = comp_df['Avg_Return'] / comp_df['Risk_StDev']

# 2. Correct Mapping Logic (Sort by Risk_StDev to define Low/Medium/High)
cluster_order = comp_df.groupby('Cluster')['Risk_StDev'].mean().sort_values().index
mapping = {
    cluster_order[0]: "Low",
    cluster_order[1]: "Medium",
    cluster_order[2]: "High"
}
comp_df['Risk_Level'] = comp_df['Cluster'].map(mapping)
print(comp_df.head())

amount=float(input("Enter the Amount you want to Invest: "))
user_cluster=input("Enter the Risk Level(Low/ Medium/ High): ").capitalize()
valid_levels = ["Low", "Medium", "High"]

if user_cluster not in valid_levels:
    print("Invalid input! Defaulting to Medium risk.")
    user_cluster = "Medium"
  
# Filter from the updated comp_df
selected_stocks = comp_df[comp_df['Risk_Level'] == user_cluster].copy()

# Rank and Allocate
selected_stocks = selected_stocks.sort_values(by='Sharpe', ascending=False)
top_stocks = selected_stocks.head(5).copy() # Get top 5 by Sharpe

weights = top_stocks['Sharpe'] / top_stocks['Sharpe'].sum()
top_stocks['Allocation'] = weights * amount

print(top_stocks[['Stock', 'Avg_Return', 'Risk_StDev', 'Sharpe', 'Allocation']])

# Plotting
plt.figure(figsize=(12,8))
plt.scatter(comp_df['Risk_StDev'], comp_df['Avg_Return'], c=comp_df['Cluster'])

for i, txt in enumerate(comp_df['Stock']):
    plt.annotate(txt, (comp_df['Risk_StDev'][i], comp_df['Avg_Return'][i]), fontsize=6)

plt.xlabel("Risk (Std Dev)")
plt.ylabel("Return")
plt.title("Risk vs Return Clustering")
plt.grid()
plt.show()