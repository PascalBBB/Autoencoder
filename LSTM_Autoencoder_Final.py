import pandas as pd
import torch
import torchmetrics as torchmetrics
from matplotlib import pyplot, pyplot as plt
from sklearn.model_selection import *
from torch.optim import Optimizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import math
import random as rn
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import torch.cuda
from torchsummary import summary

import LSTM_Autoencoder_Final_ReadData as dataClass 
import datetime
import statistics as st
from sklearn.metrics import confusion_matrix
from time import sleep
import time
from sklearn.metrics import ConfusionMatrixDisplay


SEQUENCE_LEN = dataClass.SEQUENCE_LEN
DEVICE = dataClass.device

# Writer for Tensorboard
date_time = datetime.datetime.now()
day = date_time.strftime("%d")

time_str = date_time.strftime("%H-%M-%S")
addTerm = "1-"

print(f'Seed: {dataClass.seed}')
print(f'BatchSize: {dataClass.BATCH_SIZE}')
print(f'Shuffle: {dataClass.SHUFFLE}')
run_path = '/runs/'
string = f"{run_path}-Day-{day}- Time-{time_str} + V7 - {addTerm}"


print(f'Shuffle: {dataClass.SHUFFLE}')
print(f'Seed: {dataClass.seed}')
print(f'BatchSize: {dataClass.BATCH_SIZE}')
print(f'Seq len: {dataClass.SEQUENCE_LEN}')

print(f'stored: {string}')


model_path = '/runs/'

model_train_path = model_path + "BaseLine_LSTM_Train" + "_Day-" + day + "_Time-" + time_str + ".pt"
model_train_path_dict = model_path + "BaseLine_LSTM_Train" + "_Day-" + day + "_Time-" + time_str + "_dict.pt"
model_validation_path = model_path + "BaseLine_LSTM_Validation" + "_Day-" + day + "_Time-" + time_str + "_dict.pt"


plot_save_path = '/runs/'


#print(f'Model Train Dict Stored at: {model_train_path_dict}')
#print(f'Model Validation Stored at: {model_validation_path}')
#print(f'Plot Path: {plot_save_path}')

writer = SummaryWriter(log_dir = string)
start_time = time.time()

#Dataset Daten einlesen
train_dataloader = dataClass.train_dataloader_window
validation_dataloader = dataClass.validation_dataloader_window
test_data_attacked_full_inc_normal_loader = dataClass.attacked_test_dataloader_window


INPUT_DIM = 51

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=INPUT_DIM, out_features= 48)
        self.lstm1 = nn.LSTM(input_size=48, hidden_size= 42, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=42, hidden_size= 24, batch_first=True, num_layers=1)

        self.lstm2_dec = nn.LSTM(input_size=24, hidden_size=42, batch_first=True, num_layers=1)
        self.lstm1_dec = nn.LSTM(input_size=42, hidden_size=48, batch_first=True, num_layers=1)
        self.linear1_dec = nn.Linear(in_features=48, out_features=INPUT_DIM)

        self.activation = nn.ReLU()

    def forward(self, x):
        lin1 = self.linear1(x)
        activ1 = self.activation(lin1)
        lstm1, (hl1,_) = self.lstm1(activ1)
        lstm2, (hl2,_) = self.lstm2(lstm1)
        lstm2_dec, (hl2dec,_) = self.lstm2_dec(lstm2)
        lstm1_dec, (hl1dec,_) = self.lstm1_dec(lstm2_dec)
        lin2 = self.linear1_dec(lstm1_dec)
        activ2 = self.activation(lin2)
        return lstm1[:, -1, :], lstm2[:, -1, :], lstm2_dec[:, -1, :], lstm1_dec[:, -1, :], activ2


model = Autoencoder()
model.to(device=DEVICE)
data, label = next(iter(train_dataloader))
summary(model, input_data=data.float(), verbose=2, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

#Hooks #Backward Hook for Gradientenueberpruefung
def backward_hook(module, grad_input, grad_output):
    date_time = datetime.datetime.now()
    time = date_time.strftime("%H-%M-%S")

    print(module)
    print(grad_input[0])
    print(grad_output[0])
    if grad_input[0] != None:
        input = grad_input[0].clone().detach().to('cpu')
        input = grad_input[0].clone().detach().to('cpu').numpy()
        df_i = pd.DataFrame(input)
        #df_i.to_csv(f'./files/V1LSTM/{module}_input_{time}.csv')
        writer.add_scalar(f"HookBackW/input{module}", torch.sum(grad_input[0]))
    if grad_output[0] != None:
        output = grad_output[0].clone().detach().to('cpu').numpy()
        df_o = pd.DataFrame(output)
        #df_o.to_csv(f'./files/V1LSTM/{module}_output_{time}.csv')
        writer.add_scalar(f"HookBackW/output{module}", torch.sum(grad_output[0]))
    sleep(2)

def register_backward_hook(model):
    model.lstm1.register_full_backward_hook(backward_hook)
    model.lstm2.register_full_backward_hook(backward_hook)
    model.lstm1_dec.register_full_backward_hook(backward_hook)
    model.lstm2_dec.register_full_backward_hook(backward_hook)

    
#register_backward_hook(model)


#Stochastische Dekorrelation, der Aktivierungen einer verdeckten Schicht 
def custom_loss_SDC(activ, input=None, target=None, r_para = 0.4, y_para = 10):
    """
            Initilize Random Matrix
            Multiplicated with activations -> active_dach
            Crosscovariance Matric computed
            :param activ:
            :return: Computed Stochastic Decorrelation Constraint (Value)
            """
    #Batch Size (Number of Samples)
    m = activ.size(dim=0)  
    #Init Tensor
    r = torch.ones((2,), dtype=torch.float64)
    #create Tensor with probs (Random decorrelated rate)
    r = r.new_full((activ.size()), r_para)
    #Create Random R for choose Activations for decorr
    rand_r = torch.bernoulli(r)
    #push to Device
    rand_r = rand_r.to(DEVICE)
    #Multiplication R * A = A'
    a_dach = torch.mul(rand_r, activ)

    #Sum for mean calculation
    sum = torch.sum(a_dach, dim=0)
    mü_alt = (1 / a_dach.size(dim=0)) * sum
    #Mean calculation
    mü = torch.mul((1/ a_dach.size(dim=0)), sum)

    #CrossCovariance calculation (dif. transpose first)
    cov = torch.mul((1/m), (torch.matmul(torch.transpose((a_dach-mü), 0, 1), (a_dach-mü))))

    # 2. Norm Berechnen (hier auf Vektor angewendet (doc: ganz unten vektor)
    a = torch.pow(torch.linalg.norm(torch.diagonal(cov), ord=2), 2)
    # Frobenius-Norm Berechnen (siehe doc. None = Frobenius-Norm bei Matrizen)
    b = torch.pow(torch.linalg.norm(cov, ord='fro'), 2)

    Jsdc = (y_para / 2) * (b - a)

    return Jsdc, cov, rand_r


loss_func = nn.HuberLoss()  #nn.MSELoss()

#Optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)

#Hyper-Parameter der Stochastischen Dekorrelation
R_PARAM = 0.4
Y_PARAM = 10

print(f'Activation: {model.activation.__str__()}')
print(f'Loss Function: {loss_func.__str__()}')
print(f'Optimizer: {optimizer.__str__()}')
print(f'R_Param: {R_PARAM}')
print(f'Y_Param: {Y_PARAM}')

#Kovarianzberechnung, zur Überprüfung
def calculate_cross_cov(activ):
    sum = torch.sum(activ, dim=0)
    mü = (1 / activ.size(dim=0)) * sum

    cov = (1/activ.size(dim=0)) * (torch.matmul(torch.transpose((activ-mü), 0, 1), (activ-mü)))

    summe_cov = torch.sum(cov)

    s = st.CrossCovariance()
    s.add(activ, activ)
    a = s.covariance(unbiased=False)

    summe_a = torch.sum(a)

    return summe_cov, summe_a

#Training

epochs = 30
best_valid_loss = 100000000000000
for epoch in range(epochs):
    model.train()
    loss_list = []
    loss_validation = []
    loss_mse_list = []
    jsdc_loss_list = []

    cov_hl1_list, cov_hl2_list, cov_hl2dec_list, cov_hl1dec_list = [], [], [], []

    count = 1
    with tqdm(total=len(train_dataloader)) as pbar:
        for data, label in train_dataloader:
            #torch.autograd.set_detect_anomaly(True)
            #to Device
            data = data.to(DEVICE).float()
            label = label.to(DEVICE)
            # forward pass
            hl1, hl2, hl2dec, hl1dec, reconstructed = model(data)

            # loss function MSE
            loss_mse = loss_func(reconstructed, data)
            loss_mse_list.append(loss_mse.item())
            #loss custom

            jsdc, cov, rand_r = custom_loss_SDC(activ=hl1, input=reconstructed, target=data, r_para= R_PARAM, y_para= Y_PARAM)

            loss = loss_mse #+ jsdc

            #logging
            #Cross Covariance der Hidden Layer berechnen:

            #Alle:
            hl1_encoded_copy = hl1.clone().detach()
            cov_hl1_encoded, cov_hl1_encoded_extern = calculate_cross_cov(hl1_encoded_copy)

            hl2_encoded_copy = hl2.clone().detach()
            cov_hl2_encoded, cov_hl2_encoded_extern = calculate_cross_cov(hl2_encoded_copy)

            hl2dec_decoded_copy = hl2dec.clone().detach()
            cov_hl2dec_decoded, cov_hl2dec_decoded_extern = calculate_cross_cov(hl2dec_decoded_copy)

            hl1dec_decoded_copy = hl1dec.clone().detach()
            cov_hl1dec_decoded, cov_hl1dec_decoded_extern = calculate_cross_cov(hl1dec_decoded_copy)

            writer.add_scalar("Cov_iteration/hl1_encoded_intern", cov_hl1_encoded, count)
            cov_hl1_list.append(cov_hl1_encoded.detach().to('cpu').numpy())
            writer.add_scalar("Cov_iteration/hl2_encoded_intern", cov_hl2_encoded, count)
            cov_hl2_list.append(cov_hl2_encoded.detach().to('cpu').numpy())
            writer.add_scalar("Cov_iteration/hl2dec_decoded_intern", cov_hl2dec_decoded, count)
            cov_hl2dec_list.append(cov_hl2dec_decoded.detach().to('cpu').numpy())
            writer.add_scalar("Cov_iteration/hl1dec_decoded_intern", cov_hl1dec_decoded, count)
            cov_hl1dec_list.append(cov_hl1dec_decoded.detach().to('cpu').numpy())

            h = jsdc.detach().to('cpu').numpy()
            jsdc_loss_list.append(h)

            # record loss function values
            loss_list.append(loss.item())

            #Add Wirter for Loss in iteration
            writer.add_scalar("Loss/mse_iteration", loss_mse, count)
            #writer.add_scalar("Loss/jsdc_iteration", jsdc, count)

            #clean the gradient from iteration
            optimizer.zero_grad()

            #backprob
            loss.backward()

            #gradient decent
            optimizer.step()

            count += 1

            desc = f'epoch: [{epoch + 1}/{epochs}] loss: {np.mean(loss_list):.4f} SDC: {np.mean(jsdc_loss_list)} Criterion_Loss: {np.mean(loss_mse_list)}  Cov -> hl1: {np.mean(cov_hl1_list)} | hl2:{np.mean(cov_hl2_list)} | hl2dec:{np.mean(cov_hl2dec_list)} | hl1dec:{np.mean(cov_hl1dec_list)}'
            pbar.set_description(desc)
            pbar.update()

    writer.add_scalar("Loss/train", np.mean(loss_list), epoch+1)
    writer.add_scalar("Loss/Criterion", np.mean(loss_mse_list), epoch+1) #Loss Criterion = Loss Without SDC
    writer.add_scalar("Loss/JSDC_loss", np.mean(jsdc_loss_list), epoch+1)

    writer.add_scalar("CovList/hl1encoded", np.mean(cov_hl1_list), epoch+1)
    writer.add_scalar("CovList/hl2encoded", np.mean(cov_hl2_list), epoch+1)
    writer.add_scalar("CovList/hl2decoded", np.mean(cov_hl2dec_list), epoch+1)
    writer.add_scalar("CovList/hl1decoded", np.mean(cov_hl1dec_list), epoch+1)

    model.eval() #Validierung (Validierungsdatensatz durchlaufen und bestes Modell speichern)
    with torch.no_grad():
        with tqdm(total=len(validation_dataloader)) as pbar2:
            for val_data, label in validation_dataloader:
                val_data = val_data.to(DEVICE).float()
                label = label.to(DEVICE)

                hl1, hl2, hl2dec, hl1dec, rec_valid = model(val_data)
                loss_valid = loss_func(rec_valid, val_data)
                loss_validation.append(loss_valid.item())
                #tqdm template
                desc = f'[{epoch+1}/{epochs}] --> valid_loss: {np.mean(loss_validation):.4f}'
                pbar2.set_description(desc)
                pbar2.update()
        #add to Tensorboard
        writer.add_scalar("Loss/validation_inTraining", np.mean(loss_validation), epoch+1)
        writer.add_scalars("Loss/Train-Valid", {'Loss_Train': np.mean(loss_list), 'Loss_Validation': np.mean(loss_validation), 'Loss_without_SDC': np.mean(loss_mse_list)}, epoch+1)
    if np.mean(loss_validation) < best_valid_loss:
        print(f'Best Model at {epoch+1} with {np.mean(loss_validation)}')
        best_valid_loss = np.mean(loss_validation)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_validation,
        },
            model_validation_path)


#Model save
torch.save(model, model_train_path)

#Alternativ:
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_train_path_dict)

#Model load

# model = torch.load('./models_lstm/model4_Neu_1Hidden_batch16_10Epochs.pt')
# model = model.to(dataClass.device)
#
#alternativ:
# checkpoint = torch.load(model_train_path_dict)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()


#Threshold festlegen (Schwellenwertbestimmung)
#theorie: höchste MSE der Normalen Daten
#dementsprechend der validierungsdaten des Normalen Datensatzes
reconstructed_valid_normal = torch.empty(( SEQUENCE_LEN,51), device='cpu')
orignal_validation = torch.empty((SEQUENCE_LEN,51), device='cpu')
loss_normal_val_individ = []
model = model.to('cpu')
model.eval()

with torch.no_grad():
    with tqdm(total=len(validation_dataloader)) as pbar:
        loss_validation = []
        count = 1
        count_individ = 0
        for data, label in validation_dataloader:
            data = data.to('cpu').float()

            hl1, hl2, hl2dec, hl1dec, reconstructed_val = model(data)

            for r, d in zip(reconstructed_val, data):
                count_individ += 1
                l = loss_func(r,d)
                loss_normal_val_individ.append(l)
                writer.add_scalar("Loss2/validation_data_onTrainModel", l, count_individ)

            loss_val = loss_func(reconstructed_val, data)
            loss_validation.append(loss_val.item())
            writer.add_scalar("Loss/validation", loss_val, count)
            count += 1

            desc = f'loss_valid: {np.mean(loss_validation):.4f}'
            pbar.set_description(desc)
            pbar.update()

max_valid_loss_batch = max(loss_validation)
max_valid_loss = max(loss_normal_val_individ)

print(f'max_valid_loss_batch: {max_valid_loss_batch} | max_valid_loss: {max_valid_loss}')

plt.hist(loss_normal_val_individ)
plt.title("Reconstruction Loss on Validation Data with Train Model")
plt.savefig(plot_save_path+"Reconstruction Loss on Validation Data Train Model", bbox_inches='tight')
plt.show()

#Calculate Metrics

def calculate_metrix(conf_matrix):
    print(f'Confusion Matrix: \n {conf_matrix}')
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    tn = conf_matrix[1][1]

    print(f'tp {tp}, fp: {fp}, fn: {fn}, tn: {tn}')

    total = tp + fn + fp + tn
    PP = tp + fp
    PN = fn + tn
    P = tp + fn
    N = fp + tn
    # Accuracy =  (TP + TN) / (P + N)
    acc = (tp + tn) / total

    # precision PPV = tp / (tn+ fp) oder (tp / PP)
    ppv = tp / (tp + fp)

    # Recall TP / (tp+fn) oder tp / P bzw. auch TPR
    recall = tp / (tp + fn)

    # true negativ rate = tn / (tn +fp)
    tnr = tn / (tn + fp)

    #true positive rate TPR = tp/ (tp+fn) /recall
    tpr = tp / (tp+fn)
    # false positive rate = fp / (FP+TN) oder fp / N
    fpr = fp / (fp+tn)

    # balanced accuracy = TPR + TNR /2
    bal_acc = (tpr+tnr) / 2
    #F1 Score: (2*tp) / (2tp)+fp+fn
    f1_score = (2*tp) / ((2*tp)+fp+fn)

    return f' FPR= {fpr} precision: {ppv} recall= {recall} F1_score={f1_score} -----  acc= {acc} balanced_acc= {bal_acc} TNR= {tnr}'


loss_inc_normal_in_attacked = []
loss_indiv_inc_normal_in_attacked = []
loss_batch_inc_normal_in_attacked = []
labels_test_data = []
reconstructed_inc_normal_in_attacked = torch.empty((SEQUENCE_LEN, 51), device='cpu')
model.to('cpu')

loss_label_normal = []
loss_label_attacked = []
with torch.no_grad():
    with tqdm(total=len(test_data_attacked_full_inc_normal_loader)) as pbar:
        count = 0
        count_individ = 0
        for data, labels in test_data_attacked_full_inc_normal_loader:
            data = data.to('cpu')
            count += 1

            # forward pass
            hl1, hl2, hl2dec, hl1dec, reconstructed_pred = model(data.float())

            for r, d in zip(reconstructed_pred, data):
                count_individ +=1
                l = loss_func(r,d)
                loss_indiv_inc_normal_in_attacked.append(l)
                writer.add_scalar("Loss2/attacked_data_onTrainModel", l, count_individ)

            for label in labels:
                labels_test_data.append(label)

            for label, r, d in zip(labels, reconstructed_pred, data):
                l = loss_func(r,d)
                if label == 0:
                    loss_label_normal.append(l)
                if label == 1:
                    loss_label_attacked.append(l)

            # loss function
            loss_test = loss_func(reconstructed_pred, data)
            writer.add_scalar("Loss/test_attacked_inc_normal_data", loss_test, count)
            # record loss function values
            loss_inc_normal_in_attacked.append(loss_test.item())
            loss_batch_inc_normal_in_attacked.append(loss_test)
            # tqdm template
            desc = f'loss_inc_normal_in_attacked: {np.mean(loss_inc_normal_in_attacked):.4f}'
            pbar.set_description(desc)
            pbar.update()

loss_outliers_inc_normal_in_attacked_train_model =[]
for loss_sh in loss_indiv_inc_normal_in_attacked:
    if loss_sh > max_valid_loss:
        loss_outliers_inc_normal_in_attacked_train_model.append(1)
    else:
        loss_outliers_inc_normal_in_attacked_train_model.append(0)

def plot_hist_labels(loss_label_normal, loss_label_attacked, max_valid_loss,title_plot1, title_plot2):
    #Plot Hist
    plt.hist(loss_label_normal, color="b", alpha=0.5, range=[0,max_valid_loss.item()+1.0], label="Normal")
    plt.hist(loss_label_attacked, color="r", alpha=0.6, range=[0,max_valid_loss.item()+1.0], label="Attacked")
    plt.xticks(np.arange(0,max_valid_loss.item() + 1.0, step=0.2))
    plt.title(title_plot1)
    plt.legend(loc="upper right", frameon=True)
    plt.xlabel("Loss")
    plt.savefig(plot_save_path + title_plot1, bbox_inches='tight')

    plt.show()

    plt.figure(figsize=(20,10))

    plt.hist(loss_label_normal, color="b", alpha=0.5, label="Normal")
    plt.hist(loss_label_attacked, color="r", alpha=0.6, label="Attacked")
    plt.xticks(np.arange(0,max(loss_label_attacked), step=0.2))
    plt.title(title_plot2)
    plt.legend(loc="upper right", frameon=True)
    plt.xlabel("Loss")
    plt.savefig(plot_save_path + title_plot2, bbox_inches='tight')
    plt.show()

plot_hist_labels(loss_label_normal, loss_label_attacked, max_valid_loss, title_plot1="Normal vs Attacked Reconstruction Loss in Range max Valid Loss plus 1 with Train Model", title_plot2="Normal vs Attacked Reconstruction Loss with Train Model")

count = 1
for label_thresholdBased, orig_label in zip(loss_outliers_inc_normal_in_attacked_train_model, labels_test_data):
    writer.add_scalars("Classific/OnTrainModel",{'OriginalLabel': orig_label, 'ThresholdBasedClassif': label_thresholdBased } , count)
    count +=1

#conf_matrix_inc_normal_in_attacked = confusion_matrix(y_true=dataClass.dataset_attacked_full_inc_normal.y_full_attacked_inc_normal, y_pred=loss_outliers_inc_normal_in_attacked)
conf_matrix_inc_normal_in_attacked_onTrainModel = confusion_matrix(y_true=labels_test_data, y_pred=loss_outliers_inc_normal_in_attacked_train_model, labels=[1,0])


#cm_display= ConfusionMatrixDisplay(conf_matrix_inc_normal_in_attacked_onTrainModel, display_labels=[1,0]).plot(values_format='.10g')

print(f'Combined Inc. Normal in Attacked: \n {calculate_metrix(conf_matrix_inc_normal_in_attacked_onTrainModel)}')



#Eigentliche Methode, die auf dem Trainingsmodell dient nur der Ueberprüfung
print("On Validation:----------------------------------------------------------------------------------")
checkpoint = torch.load(model_validation_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()


#MaxValidLoss
reconstructed_valid_normal = torch.empty(( SEQUENCE_LEN,51), device='cpu')
orignal_validation = torch.empty((SEQUENCE_LEN,51), device='cpu')

loss_normal_val_individ = []
model = model.to('cpu')
with torch.no_grad():
    with tqdm(total=len(validation_dataloader)) as pbar:
        loss_validation = []
        count = 1
        count_individ = 0
        for data, labels in validation_dataloader:
            data = data.to('cpu').float()


            hl1, hl2, hl2dec, hl1dec, reconstructed_val = model(data)


            for r, d in zip(reconstructed_val, data):
                count_individ += 1
                l = loss_func(r,d)
                loss_normal_val_individ.append(loss_func(r, d))
                writer.add_scalar("Loss2/validation_data_onValidModel", l, count_individ)

            loss_val = loss_func(reconstructed_val, data)
            loss_validation.append(loss_val.item())
            writer.add_scalar("Loss/validation_validModel", loss_val, count)
            count += 1

            desc = f'loss_valid: {np.mean(loss_validation):.4f}'
            pbar.set_description(desc)
            pbar.update()

max_valid_loss_batch = max(loss_validation)
max_valid_loss = max(loss_normal_val_individ)

print(f'max_valid_loss_batch: {max_valid_loss_batch} | max_valid_loss: {max_valid_loss}')
plt.hist(loss_normal_val_individ)
plt.title("Reconstruction Loss on Validation Data with Validation Model")
plt.show()

#Test on Testdata
loss_inc_normal_in_attacked = []
loss_indiv_inc_normal_in_attacked = []
loss_batch_inc_normal_in_attacked = []
labels_test_data_on_valid_model = []
reconstructed_inc_normal_in_attacked = torch.empty((SEQUENCE_LEN, 51), device='cpu')
model.to('cpu')
loss_label_normal, loss_label_attacked = [], []
with torch.no_grad():
    with tqdm(total=len(test_data_attacked_full_inc_normal_loader)) as pbar:
        count = 0
        count_indiv = 0
        for data, labels in test_data_attacked_full_inc_normal_loader:
            data = data.to('cpu')
            count += 1

            # forward pass
            hl1, hl2, hl2dec, hl1dec, reconstructed_pred = model(data.float())

            for r, d in zip(reconstructed_pred, data):
                count_indiv += 1
                l = loss_func(r,d)
                loss_indiv_inc_normal_in_attacked.append(l)
                writer.add_scalar("Loss2/test_attacked_individ_onValidModel", l, count_indiv)


            for label in labels:
                labels_test_data_on_valid_model.append(label)
            # loss function
            loss_test = loss_func(reconstructed_pred, data)
            writer.add_scalar("Loss/test_attacked_inc_normal_data_validModel", loss_test, count)

            for label, r, d in zip(labels, reconstructed_pred, data):
                l = loss_func(r,d)
                if label == 0:
                    loss_label_normal.append(l)
                if label == 1:
                    loss_label_attacked.append(l)

            # record loss function values
            loss_inc_normal_in_attacked.append(loss_test.item())
            loss_batch_inc_normal_in_attacked.append(loss_test)
            # tqdm
            desc = f'loss_inc_normal_in_attacked: {np.mean(loss_inc_normal_in_attacked):.4f}'
            pbar.set_description(desc)
            pbar.update()

plot_hist_labels(loss_label_normal, loss_label_attacked, max_valid_loss,
                 title_plot1="Normal vs Attacked Reconstruction Loss with Validation Model in Range max Valid Loss plus 1",
                 title_plot2="Normal vs Attacked Reconstruction Loss with Validation Model")


loss_outliers_inc_normal_in_attacked =[]
for loss_sh in loss_indiv_inc_normal_in_attacked:
    if loss_sh > max_valid_loss:
        loss_outliers_inc_normal_in_attacked.append(1)
    else:
        loss_outliers_inc_normal_in_attacked.append(0)

#Alternativ für Plots
count = 1
for label_thresholdBased, orig_label in zip(loss_outliers_inc_normal_in_attacked, labels_test_data_on_valid_model):
    writer.add_scalars("Classific/OnValidationModel",{'OriginalLabel': orig_label, 'ThresholdBasedClassif': label_thresholdBased } , count)
    count +=1


#conf_matrix_inc_normal_in_attacked = confusion_matrix(y_true=dataClass.dataset_attacked_full_inc_normal.y_full_attacked_inc_normal, y_pred=loss_outliers_inc_normal_in_attacked)
conf_matrix_inc_normal_in_attacked = confusion_matrix(y_true=labels_test_data_on_valid_model, y_pred=loss_outliers_inc_normal_in_attacked, labels=[1,0])
print(f'Combined Inc. Normal in Attacked: \n {calculate_metrix(conf_matrix_inc_normal_in_attacked)}')



####Show Distribution of FP Classifications over Time
#Labels on TestData vs. Outlier List

#plt.plot(labels_test_data_on_valid_model)

#Plot FP Classifications
plt.figure(figsize=(120,10), dpi=80)
plt.plot(labels_test_data, color="blue", linestyle="-", label="Original")
plt.plot(loss_outliers_inc_normal_in_attacked_train_model, color= "red", linestyle="--", label="Threshold Based Classification")
plt.legend(loc="upper right", frameon=True)
plt.title("Original label vs. Threshold Based Classification over Time on Train Model")
plt.xlabel("Time")
plt.xticks(np.arange(0,450000, 10000), rotation=45)
plt.savefig(plot_save_path + "Original label vs Threshold Based Classification over Time on Train Model", bbox_inches='tight')
plt.show()

###Labels on TestData vs. Outlier List Model Validation
plt.figure(figsize=(120,10), dpi=80)
plt.plot(labels_test_data_on_valid_model, color="blue", linestyle="-", label="Original")
plt.plot(loss_outliers_inc_normal_in_attacked, color= "red", linestyle="--", label="Threshold Based Classification")
plt.legend(loc="upper right", frameon=True)
plt.title("Original label vs. Threshold Based Classification over Time on Best Validation Model")
plt.xlabel("Time")
plt.xticks(np.arange(0,450000, 10000), rotation=45)
plt.savefig(plot_save_path + "Original label vs Threshold Based Classification over Time on Best Validation Model", bbox_inches='tight')
plt.show()


writer.flush()
writer.close()
end_time = time.time()
print(f'Ausführungsdauer: {end_time-start_time}, in min: {(end_time-start_time)/60} in std: {(((end_time-start_time)/60)/60)}')

print("done")
