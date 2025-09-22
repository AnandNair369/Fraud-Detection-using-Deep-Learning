#By Anand Nair
# Date: 27th August 2025
# # Fraud detection with GNN


import kagglehub
ealtman2019_credit_card_transactions_path = kagglehub.dataset_download('ealtman2019/credit-card-transactions')

print('Data source import complete.')

# Necessary imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

#Preprocessing the data

# Load the tabular dataset
df = pd.read_csv("/kaggle/input/credit-card-transactions/credit_card_transactions-ibm_v2.csv").sample(n=100000, random_state=42)

df.info()


df [df['Is Fraud?'] == 'Yes'].shape

percent_missing=(df.isnull().sum()*100/df.shape[0]).sort_values(ascending=True)
plt.title("Missing Value Analysis")
plt.xlabel("Features")
plt.ylabel("% of missing values")
plt.bar(percent_missing.sort_values(ascending=False).index,percent_missing.sort_values(ascending=False),color=(0.1, 0.1, 0.1, 0.1),edgecolor='blue')
plt.xticks(rotation=90)


df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)

df.Amount.head(5)

# Strip the ‘$’ from the Amount to cast as a float
df["Amount"]=df["Amount"].str.replace("$","").astype(float)

# Time can't be casted to int so opted to extract the hour and minute
df["Hour"] = df["Time"].str [0:2]
df["Minute"] = df["Time"].str [3:5]

df.Hour

df.Minute

df = df.drop(["Time","User","Card"],axis=1)
df.info()

df["Errors?"].unique()

df["Errors?"]= df["Errors?"].fillna("No error")

df = df.drop(columns=["Merchant State","Zip"],axis=1)

# change the is fraud column to binary
df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)

df["Merchant City"]=LabelEncoder().fit_transform(df["Merchant City"])
df["Use Chip"]=LabelEncoder().fit_transform(df["Use Chip"])
df["Errors?"]=LabelEncoder().fit_transform(df["Errors?"])

df["Errors?"].unique()

df.info()



# Create an empty graph
G = nx.MultiGraph()

# Add nodes to the graph for each unique card_id, merchant_name
G.add_nodes_from(df["card_id"].unique(), type='card_id')
G.add_nodes_from(df["Merchant Name"].unique(), type='merchant_name')


# Add edges and properties to the edges
for _, row in df.iterrows():
    # Create a variable for each properties for each edge

        year = row["Year"],
        month = row["Month"],
        day = row["Day"],
        hour = row["Hour"],
        minute =row["Minute"],
        amount = row["Amount"],
        use_chip =  row["Use Chip"],
        merchant_city = row["Merchant City"],
        errors =  row["Errors?"],
        mcc = row['MCC']


        G.add_edge(row['card_id'], row['Merchant Name'], year = year , month = month , day = day ,
              hour = hour , minute = minute , amount = amount , use_chip = use_chip ,
              merchant_city = merchant_city , errors = errors , mcc = mcc)

# Get the number of nodes and edges in the graph
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Print the number of nodes and edges
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)

# Convert the graph to an adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()

adj_matrix.shape

# Get a small sample of the nodes in the graph
sample_nodes = list(G.nodes())[:10]

# Retrieve the properties of the sample nodes
node_properties = nx.get_node_attributes(G, 'type')

# Print the properties of the sample nodes
for node in sample_nodes:
    print(f"Node: {node}, Properties: {node_properties[node]}")

sample_size = 5
for i, edge in enumerate(G.edges()):
    print(G.get_edge_data(*edge))
    if i >= sample_size - 1:
        break


# Retrieve the properties errors of all the edges
edge_properties = nx.get_edge_attributes(G, 'errors')

# Count the number of edges by property value
edge_count_by_property = Counter(edge_properties.values())

# Print the count of edges by property value
for property_value, count in edge_count_by_property.items():
    print(f"Property value: {property_value}, Count: {count}")

# Prepare the data for input into the model
edge_list = list(G.edges(data=True))

list(edge_list[i][2].values())


class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FraudGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

# Prepare the data for input into the model
edge_list = list(G.edges(data=True))
x = []
for edge in edge_list:
    edge_values = list(edge[2].values())
    edge_values = [float(i[0]) if type(i) == tuple and type(i[0]) == str else i[0] if type(i) == tuple else i for i in edge_values]
    x.append(edge_values)
x = torch.tensor(x, dtype=torch.float)

target = torch.tensor(df['Is Fraud?'].values, dtype=torch.float)

# Define the model
input_dim = len(x[0])
hidden_dim = 16
model = FraudGNN(input_dim, hidden_dim)
num_epochs=201

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for i in range(num_epochs):
    # Forward pass
    output = model(x)
    # Compute the loss
    loss = criterion(output, target)
    if i % 20 == 0:
        print(f'Epoch: {i}, Loss: {loss.item()}')
    # Zero the gradients
    optimizer.zero_grad()
    # Perform backpropagation
    loss.backward()
    # Update the parameters
    optimizer.step()


# necessary imports of this part
from inductiveGRL.graphconstruction import GraphConstruction
from inductiveGRL.hinsage import HinSAGE_Representation_Learner

# Global parameters:
embedding_size = 64
add_additional_data = True



# we will take 70% of our dataset as traindata
cutoff = round(0.7*len(df))
train_data = df.head(cutoff)
inductive_data = df.tail(len(df)-cutoff)

print('The distribution of fraud for the train data is:\n', train_data['Is Fraud?'].value_counts())
print('The distribution of fraud for the inductive data is:\n', inductive_data['Is Fraud?'].value_counts())


transaction_node_data = train_data.drop("card_id", axis=1).drop("Merchant Name", axis=1).drop('Is Fraud?', axis=1)
client_node_data = pd.DataFrame([1]*len(train_data["card_id"].unique())).set_index(train_data["card_id"].unique())
merchant_node_data = pd.DataFrame([1]*len(train_data["Merchant Name"].unique())).set_index(train_data["Merchant Name"].unique())

nodes = {"client":train_data.card_id, "merchant":train_data["Merchant Name"], "transaction":train_data.index}
edges = [zip(train_data.card_id, train_data.index),zip(train_data["Merchant Name"], train_data.index)]
features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

graph = GraphConstruction(nodes, edges, features)
S = graph.get_stellargraph()
print(S.info())

#GraphSAGE parameters
num_samples = [2,32]
embedding_node_type = "transaction"

hinsage = HinSAGE_Representation_Learner(embedding_size, num_samples, embedding_node_type)
trained_hinsage_model, train_emb = hinsage.train_hinsage(S, list(train_data.index), train_data['Is Fraud?'], batch_size=5, epochs=10)



"""

IBM dataset :
* [Link1](https://github.com/IBM/TabFormer) github
* [Link2](https://ibm.ent.box.com/v/tabformer-data) IBM@Box
* [Link3](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions) kaggle

"""