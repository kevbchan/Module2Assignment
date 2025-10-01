import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("CarsDatasets2025.csv", encoding='ISO-8859-1')
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")

def parse_price(price_str):
    if pd.isna(price_str):
        return None
    price_str = str(price_str).replace('$','').replace(',','').strip()
    if price_str.lower() in ['n/a', 'none', 'na', 'n']:
        return None
    if '-' in price_str:
        try:
            return float(price_str.split('-')[0].strip())
        except:
            return None
    elif '/' in price_str:
        try:
            return float(price_str.split('/')[0].strip())
        except:
            return None
    else:
        try:
            return float(price_str)
        except:
            return None

df['Price_num'] = df['Cars Prices'].apply(parse_price)

engine_filter = ['I4', 'Inline-4']
df_filtered = df[
    (df['Price_num'].notna()) &
    (df['Price_num'] < 40000) &
    (df['Engines'].isin(engine_filter))
].reset_index(drop=True)

df_filtered = df_filtered.drop_duplicates(subset=['Cars Names'])
print(f"Filtered cars under $40k with 4-cylinder engines (duplicates removed): {len(df_filtered)} rows")

def parse_numeric(val):
    if pd.isna(val):
        return 0
    val = str(val).split()[0].replace(',','').strip()
    try:
        return float(val)
    except:
        return 0

df_filtered['HorsePower_num'] = df_filtered['HorsePower'].apply(parse_numeric)
df_filtered['TotalSpeed_num'] = df_filtered['Total Speed'].apply(parse_numeric)
df_filtered['Seats_num'] = df_filtered['Seats'].apply(parse_numeric)
df_filtered['Torque_num'] = df_filtered['Torque'].apply(parse_numeric)
df_filtered['CC_num'] = df_filtered['CC/Battery Capacity'].apply(parse_numeric)

features = ['HorsePower_num', 'TotalSpeed_num', 'Seats_num', 'Torque_num', 'CC_num', 'Price_num']
X = df_filtered[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sim_matrix = cosine_similarity(X_scaled)

G = nx.Graph()
for i, car in enumerate(df_filtered['Cars Names']):
    G.add_node(car)

threshold = 0.92
for i in range(len(df_filtered)):
    for j in range(i+1, len(df_filtered)):
        if sim_matrix[i, j] > threshold:
            G.add_edge(df_filtered['Cars Names'][i], df_filtered['Cars Names'][j], weight=sim_matrix[i,j])

deg_centrality = nx.degree_centrality(G)
top_cars = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
print("Top 15 Most Similar Cars under $40k (by degree centrality):")
for car, score in top_cars:
    print(f"{car}: {score:.3f}")

top_nodes = [car for car, _ in top_cars]
G_sub = G.subgraph(top_nodes)

node_sizes = [500 + 2000 * deg_centrality[node] for node in G_sub.nodes()]
top3_nodes = top_nodes[:3]
node_colors = ['orange' if node in top3_nodes else 'skyblue' for node in G_sub.nodes()]

plt.figure(figsize=(12,12))
pos = nx.kamada_kawai_layout(G_sub)
nx.draw(
    G_sub, pos, 
    with_labels=True, 
    node_size=node_sizes, 
    node_color=node_colors, 
    font_size=10, 
    font_weight='bold', 
    edge_color='gray'
)
plt.title("Top 15 Most Similar 4-Cylinder Cars under $40k", fontsize=16)
plt.show()
