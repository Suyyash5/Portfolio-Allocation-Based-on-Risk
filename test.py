import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 1. List of your file paths
files = [
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ADANI_ENTERPRISES.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ADANI_PORTS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\APOLLO HOSPITALS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ASIAN PAINTS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\AXIS_BANK.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BAJAJ AUTO.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BAJAJ_FINANCE.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BAJAJ_FINSERV.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BHARAT PETROLEUM.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BHARTI_AIRTEL.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\BRITANNIA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\CIPLA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\COAL INDIA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\DIVIS LAB.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\EICHER MOTOTRS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\GRASIM.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HCL_TECHNOLOIES.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HDFC_BANK.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HDFC_LIFE.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HERO MOTOCORP.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HINDALCO.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\HINDUSTAN UNILEVER.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ICICI_BANK.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\INDUS INDUSTRIES.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\INFOSYS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ITC.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\JSW STEEL.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\KOTAK_MAHINDRA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\MARUTI SUZUKI.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\NESTLE.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\NTPC.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ONGC.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\POWERGRID.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\RELIANCE.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\SBI_BANK.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\SBI_LIFE.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\SUN_PHARMA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TATA CONSULTANCY SERVICES.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TATA CONSUMER PRODUCTS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TATA MOTORS.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TATA STEEL.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TECH_MAHINDRA.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\TITAN.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\ULTRATECH CEMENT.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\UPL.csv",
    r"C:\Users\HP\OneDrive\Documents\Semester 4\Project\NIFTY 50\WIPRO.csv",
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