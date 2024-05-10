import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import numpy as np

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import glob
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model).double()
        self.W_k = nn.Linear(d_model, d_model).double()
        self.W_v = nn.Linear(d_model, d_model).double()
        self.W_o = nn.Linear(d_model, d_model).double()
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff).double()
        self.fc2 = nn.Linear(d_ff, d_model).double()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.norm3 = nn.LayerNorm(d_model).double()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_output, src_mask):
        attn_output = self.self_attn(enc_output, enc_output, enc_output, src_mask)
        x = enc_output
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, d_output, seq_len):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(100, d_model)
        #Création des encoders et decoder au nombre de num_layers chacun
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.d_model = d_model
        self.seq_len = seq_len
        #Creation de la couche linéaire finale
        self.fc = nn.Linear(d_model, d_output).double()
        #Creation du dropout
        #self.dropout = nn.Dropout(dropout) -------------A voir-------------

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        batch_size, d_model, _, seq_len, _ = src_mask.size()
        if d_model==1:
            src_mask = src_mask.view(batch_size, d_model, d_model, seq_len)
        #src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src))) pas besoin deja embedded; on change un peu la fonction

        #-----------------------------Utilisation de dropout-----------------------------
        #src_embedded = self.dropout(src)
        #-----------------------------Utilisation de dropout-----------------------------
        
        #On créé des valeur inutile pour garder la structure de base si on besoin d'une adaptation avec dropout
        src_embedded = src

        #Couche des N encodeurs
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        #Couche des N decodeurs
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(enc_output, src_mask)
        
        #Sortie linéaire
        output = self.fc(dec_output)
        return output
    
    def one_epoch(self, j, X_input, Y, batch_size, optimizer, criterion):
        input = torch.from_numpy(X_input[j:j+batch_size]).view(batch_size, self.seq_len, self.d_model)
        labels = torch.from_numpy(Y[j:j+batch_size]).view(batch_size)
        optimizer.zero_grad()
        
        output = self(input)
        loss = criterion(output[:,0,:], labels)
        loss.backward()

        optimizer.step()
        return loss.item()
    
    def train_model(self, device, X_input, Y, batch_size, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0005)  #RTIDS
        len_x = np.shape(X_input)[0] 
        if batch_size>len_x:
            batch_size=len_x
        for epoch in range(num_epochs):
            self.train(True)
            running_loss = 0.
            len_without_rest = len_x - len_x%batch_size
            for j in tqdm(range(0, len_without_rest, batch_size)):
                running_loss += self.one_epoch(j, X_input.to(device), Y.to(device), batch_size, optimizer, criterion)
            #On fait la vision euclidienne car le dernier batch n'est pas forcément pile de la longeur du batch voulue (plus petit)
            if len_x%batch_size!=0:
                running_loss += self.one_epoch(j, X_input, Y, len_x%batch_size, optimizer, criterion)
            print(f"Epoch: {epoch+1}, Loss: {running_loss}")
    
    #def predict(self, X_input):



class Flux:
    #Lors de la création d'un nouveau flux, on créé un matrice vide
    def __init__(self, source_port, dest_port, packet, d_model, d_historique):
        self.sp = source_port
        self.dp = dest_port
        matrice_base = np.zeros((d_model, d_historique-1))
        self.matrice = np.insert(matrice_base,0, packet, axis=1)

    def decalage_matriciel(self, vecteur):
        #d_historique = np.shape(self.matrice)[1]
        #On s'assure que la matrice et le vecteur sont de la bonne taille
        #assert d_model == np.shape(self.matrice)[0]
        #Décalage
        self.matrice = self.matrice[:,0:d_historique-1]
        #Ajout de la nouvelle colonne
        self.matrice = np.insert(self.matrice,0, vecteur, axis=1)

class Flux_protocol:
    #Lors de la création d'un nouveau flux, on créé un matrice vide
    def __init__(self, protocol, packet, d_model, d_historique):
        self.protocol = protocol
        matrice_base = np.zeros((d_model, d_historique-1))
        self.matrice = np.insert(matrice_base,0, packet, axis=1)

    def decalage_matriciel(self, vecteur):
        #d_historique = np.shape(self.matrice)[1]
        #On s'assure que la matrice et le vecteur sont de la bonne taille
        #assert d_model == np.shape(self.matrice)[0]
        #Décalage
        self.matrice = self.matrice[:,0:d_historique-1]
        #Ajout de la nouvelle colonne
        self.matrice = np.insert(self.matrice,0, vecteur, axis=1)

def flow_exists_and_append(flows, data_input, packet, source_port, dest_port):
    for flow in flows:
        sp = int(flow.sp)
        dp = int(flow.dp)
        #On teste si il existe déjà un flux connu
        if ((sp==source_port and dp==dest_port) or (sp==dest_port and dp==source_port)):
            #Si trouvé, on ajoute le nouveau vecteur en décalant la matrice
            flow.decalage_matriciel(packet)
            #On ajoute par la même occasion la matrice aux données entrante de notre modèle
            data_input.append(flow.matrice)
            return
    #Si aucun flux n'a été trouvé, on ajoute un nouveau
    flows.append(Flux(source_port, dest_port, packet, d_model, d_historique))
    #flows[len(flows)-1].decalage_matriciel(packet)
    #On ajoute la matrice correspondante à ce flux aux données entrantes
    data_input.append(flows[len(flows)-1].matrice)


def importation_csv():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('./TrafficLabelling/*.csv')

    # Create an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each CSV file and append its contents to the combined dataframe
    for csv_file in csv_files:
        print(csv_file.title)
        df = pd.read_csv(csv_file, encoding='cp1252')
        combined_df = pd.concat([combined_df, df])
    data = combined_df

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

def creation_X_Y_ip(data):
    # On retire la première ligne (headers) et les colonnes donc on ne se sert pas ( 'Flow ID' ' Source IP' ' Source Port' ' Destination IP' ' Destination Port' ' Protocol' ' Timestamp' 'Label')
    # On mettra séparement les colonnes port source et dest et on ajoutera le protocol en onehot encoding
    data_without_nan = data.values[~pd.isna(data.values).any(axis=1)]
    source_port = np.vstack(data_without_nan[1:,2])
    dest_port = np.vstack(data_without_nan[1:,4])
    protocol = np.vstack(data_without_nan[1:,5])

    source_ip = np.vstack(data_without_nan[1:,1])
    dest_ip = np.vstack(data_without_nan[1:,3])
    X = data_without_nan[1:,7:]
    #Conversion One Hot Encoding de la colonne Protocol
    ohe = data_without_nan[1:,5]
    #print(np.shape(ohe))
    #print(ohe)
    #ohe = ohe.replace(np.nan, 0)
    #print(np.shape(ohe))
    ohe = pd.get_dummies(ohe.astype(int), dtype=int)
    Y = np.vstack(X[:,-1])
    Y = np.array(Y)
    X = X[:,:np.shape(X)[1]-1]
    #On ajoute ces colonnes aux précédentes
    X = minmax_scale(X, axis=0)
    X = np.concatenate((source_port, X), axis=1)
    X = np.concatenate((dest_port, X), axis=1)
    X = np.concatenate((ohe.values, X), axis=1)
    return X, Y, source_ip, dest_ip, protocol

def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test = train_test_split(X,Y,source_ip,dest_ip,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test

def transformation_2D(X, source_ip, dest_ip):
    # Liste des flux (sauvegarde temporaire)
    flows = []
    # Données en entrée du modèle
    data_input = []
    d_model = np.shape(X)[1]
    i=0

    for raw in tqdm(X):
        #print("Raw:", raw)
        flow_exists_and_append(flows, data_input, raw, source_ip[i],dest_ip[i])
        i =i +1

    return(np.array(data_input))


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return np.transpose(P)


def split_npy_save(array, number_of_files, folder):
    file_name = 'X_input'
    i=0
    mem = 0
    len_array = len(array)
    for i in range(number_of_files):
        np.save('./'+folder+'/X_input_'+str(i)+'.npy', array[mem:int((i+1)*len_array/number_of_files)])
        mem =int((i+1)*len_array/number_of_files)


print("--------------------Importation données--------------------")
data_frame = importation_csv()
print("--------------------Séparation des données--------------------")
X_data, Y_data, source_ip_data, dest_ip_data, protocol = creation_X_Y_ip(data_frame)


#Choix des données pour l'entrainement du modèle
print("--------------------Sélection des données d'entrainement--------------------")
X_input, X_test, Y, Y_test, source_ip, source_ip_test, dest_ip, dest_ip_test = choix_donnees_entrainement_70_30(X_data, Y_data, source_ip_data, dest_ip_data)
print("--------------------Création des tableaux 2D pour les données entrainement--------------------")

d_output = 15 #Nombre de labels
d_model = 1
seq_len = np.shape(X_input)[1] #np.shape(X_input)[1] #Longueur du vecteur d'entrée (d_model) normalement 82
num_heads = 1  #d_model % num_heads == 0, "d_model must be divisible by num_heads"
num_layers = 6 #RTIDS Nombre de répétition des encoders/decoders
d_ff = 1024 #RTIDS dimension du FFN layer
dropout = 0.5 #RTIDS
batch_size = 128 #RTIDS batch_size = 128
epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformer = Transformer(d_model, num_heads, num_layers, d_ff, dropout, d_output, seq_len)
transformer.to(device)
transformer.train_model(X_input, Y, batch_size, epochs)