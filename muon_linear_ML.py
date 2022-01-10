import ROOT as root
import numpy as np
import torch
import os
import optuna
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from optuna import Trial as trial
import joblib

#################################################################

#  this macro Trained the Model for the best hyper parameters

#################################################################

# use the multi GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# check the GPU
print("=====================================")
print("Avliable to GPU : ", torch.cuda.is_available())
print('cuda index:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print("=====================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# option
file_save =1
model_save =0# option
file_save =1
model_save =0
tuneVersion = "3d_layer2_2"

Total_epoch = 30
n_trials = 40






class MuonPadDataset(Dataset):
  def __init__(self):
    self.x_data = np.load('./dataTPC/array/3d/trainPad.npy')
    self.y_data = np.load('./dataTPC/array/3d/trainPadTrack.npy')

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx]).to(device)
    y = torch.FloatTensor(self.y_data[idx]).to(device)
    return x, y



class MuonPadDataSetTest(Dataset):
  def __init__(self):
    self.x_data = np.load('./dataTPC/array/3d/validationPad.npy')
    self.y_data = np.load('./dataTPC/array/3d/validationPadTrack.npy')

  def __len__(self):
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx]).to(device)
    y = torch.FloatTensor(self.y_data[idx]).to(device)


class TrackingModel(torch.nn.Module):
    def __init__(self, conv1, conv2, drop1, drop2, l1, l2):
        super(TrackingModel, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, conv1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(conv1, conv2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.layer3 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop1))

        self.fc2 = torch.nn.Linear(l1, l2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop2))

        self.fc3 = torch.nn.Linear(l2, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)



    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out


        'epoch': Total_epoch ,
        'conv1': trial.suggest_categorical('conv1', [5, 10, 25, 50, 75, 100]),
        'conv2': trial.suggest_categorical('conv2', [30, 50, 80, 100, 150, 200]),
        'drop1': trial.suggest_uniform('drop1', 0.3, 0.6),
        'lr': trial.suggest_loguniform('lr', 0.000005, 0.0003),
        'batchSize': trial.suggest_categorical('batchSize', [64, 128, 256, 512, 1024])
    DataSetPadTest = MuonPadDataSetTest()
    DataPad = DataLoader(DataSetPad, batch_size=int(config['batchSize']), shuffle=True, drop_last=True)
    DataPadvaild = DataLoader(DataSetPadTest, batch_size=int(config['batchSize']), shuffle=True, drop_last=True)

    model = nn.DataParallel(_model).to(device)  ## multi gpu

    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4,)


    print("=====================================")
    print("current trial number: ", trial.number)
    print("=====================================")

    valid_loss =0
    for epoch in range(config['epoch']):

        print("------------ epoch : ", epoch ,"---------------")

        # train loop
        train_loss = 0
        trainSize = len(DataPad.dataset)
        for batch, (X, y) in enumerate(DataPad ,0):


            pred = model(X)
            loss = loss_func(pred, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            train_loss = loss
            loss, current = loss.item(), batch * len(X)
            if(batch%100 == 0):
                print(f"train loss: {loss:>7f}  data[{current:>5d}/{trainSize:>5d}]")


        # vaildation loop
        fit = root.TF1("fit","gaus",-5,5)
        fit2 = root.TF1("fit","gaus",-5,5)

        h1 = root.TH1D("Pad","",1000,-3, 3)
        h2 = root.TH1D("Drift","",1000,-3, 3)
        test_loss = 0
        num_batches = len(DataPadvaild)


        with torch.no_grad():
            for X, y in DataPadvaild:

                model.eval()
                pred = model(X)
                test_loss += loss_func(pred, y).item()

                for i in range(len(pred)):
                    z0 = pred[i][2]*150 - y[i][2]*150
                    z100 = pred[i][3]*150 - y[i][3]*150

                    h2.Fill(z0)
        mean2 = fit2.GetParameter(1)

        print("-------------     Validation    ------------------")
        print(f"Avg loss: {test_loss:>8f} \n")

        if(file_save==1):
            f = open("./par_optim_Info/tuneParInfo_v{}.txt".format(tuneVersion), 'a')
            data =''
            if(epoch == 0):
            else:

            f.write(data)
            f.close()

        del h1
        del h2
        del fit
        del fit2

        trial.report(valid_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    if(model_save ==1):
        torch.save(model, './model/bestModel_v{}.pth'.format(tuneVersion))


    return valid_loss

if __name__ == '__main__':
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="minimize")
    study.optimize(train_func, n_trials=n_trials)

    if(file_save==1):
        joblib.dump(study, './par_optim_Info/tuneResult_v{}.pkl'.format(tuneVersion))

    study.trials
