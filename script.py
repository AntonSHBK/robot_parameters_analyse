import torch
import pandas as pd
import numpy as np
from typing import List
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def load_data(path: Path):
    print('**********************************')
    print("Read data set from path {path}".format(path=path))
    df = pd.read_csv(path)
    df["constraint1"] = df["constraint1"].astype('bool')
    df["constraint2"] = df["constraint2"].astype('bool')
    return df

data_1_path = Path('data/DataSet900.csv')

dataset = load_data(data_1_path)
print('lol')
# class MyDataset(Dataset):
#     def __init__(self, 
#                  data:pd.DataFrame,
#                  input_colums: List[str],
#                  output_colums: List[str]) -> None:
#         self.input_colums = input_colums
#         self.output_colums = output_colums
        
#         x = data[self.input_colums].values
#         y = data[self.output_colums].values
        
#         self.x_train = torch.tensor(x, dtype=torch.float32)
#         self.y_train = torch.tensor(y, dtype=torch.float32)      
        
#     def __len__(self):
#         return len(self.y_train)
    
#     def __getitem__(self, index):
#         return self.x_train[index], self.y_train[index]   

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self, inputs, outputs, middle_layers=128):
#         super(NeuralNetwork, self).__init__()
#         self.linear_relu_stack = nn.Sequential(            
#             nn.Linear(inputs, middle_layers),
#             nn.ReLU(),
#             nn.Linear(middle_layers, middle_layers),
#             nn.ReLU(),
#             nn.Linear(middle_layers, outputs)
#         )

#     def forward(self, x: torch.Tensor):
#         logits = self.linear_relu_stack(x)
#         return logits

# test = NeuralNetwork(2,2).to(device)
# print(test)
# from numpy import mean
# from torch.nn import MSELoss
# from torch.optim import SGD

# from sklearn.model_selection import train_test_split


# class Trainer():
    
#     def __init__(self,
#                  data: pd.DataFrame,
#                  input_colums: List[str],
#                  output_colums: List[str],
#                  batch_size=32,
#                  learning_rate=0.005,
#                  shuffle=False,
#                  test_size=0.2,
#                  random_state=None,
#                  save_path='saved_model/model_weights.pth'):
        
#         self.save_path = save_path
        
#         self.input_colums = input_colums
#         self.output_colums = output_colums
        
#         self.normalize_param = {}
#         self._normalize_data(data, input_colums+output_colums)       

#         train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
                
#         self.train_dataloader = DataLoader(
#             MyDataset(train_set, input_colums, output_colums),
#             batch_size=batch_size,
#             shuffle=shuffle
#         )
#         self.test_dataloader = DataLoader(
#             MyDataset(test_set, input_colums, output_colums),
#             batch_size=batch_size,
#             shuffle=False
#         )
        
#         self.model = NeuralNetwork(
#             inputs=len(input_colums),
#             outputs=len(output_colums),
#             middle_layers=512
#         ).to(device)
        
#         self.loss_func = MSELoss()
#         self.optimizer = SGD(self.model.parameters(), lr=learning_rate)       
    
#     def train(self):
#         size = len(self.train_dataloader.dataset)
#         self.model.train()
#         for batch, (X, y) in enumerate(self.train_dataloader):
#             X, y = X.to(device), y.to(device)

#             # Compute prediction error
#             pred = self.model(X)
#             loss: torch.Tensor = self.loss_func(pred, y)

#             # Backpropagation
#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()

#             if batch % 10 == 0:
#                 loss, current = loss.item(), (batch + 1) * len(X)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
#     def test(self):
#         num_batches = len(self.test_dataloader)
#         self.model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for X, y in self.test_dataloader:
#                 X, y = X.to(device), y.to(device)
#                 pred = self.model(X)
#                 test_loss += self.loss_func(pred, y).item()
#         test_loss /= num_batches
#         print(f"Test Error: Avg loss: {test_loss:>8f} \n")
        
#     def run(self,  epochs=10):
#         for t in range(epochs):
#             print(f"Epoch {t+1}\n-------------------------------")
#             self.train()
#             self.test()
#         print("Done!")
    
#     def predict(self, input: np.array):
#         print('Входные данные', input)
#         for index, item in enumerate(self.input_colums):
#             input[index] = (input[index] - self.normalize_param[item]['mean']) /\
#                 self.normalize_param[item]['std']
#         print('Нормализованные', input)
#         pred:torch.Tensor = self.model(torch.tensor(input, dtype=torch.float32))        
#         pred = pred.detach().numpy()
#         print('Предикт модели', pred)
#         for index, item in enumerate(self.output_colums):
#             # pred[index] = pred[index] * self.normalize_param[item]['lenght']
#             pred[index] = pred[index] * self.normalize_param[item]['std'] +\
#                 self.normalize_param[item]['mean']
#         print('Денормализованные данные', pred)
#         return pred
        
    
#     def _normalize(self,
#                    data: pd.DataFrame,
#                    colum: str) -> torch.Tensor:
#         mean = data[colum].mean()
#         std = data[colum].std()
#         data[colum] = (data[colum] - mean) / std  
#         return {'mean': mean, 'std': std}
#         # lenght = data[colum].max() - data[colum].min()
#         # data[colum] = data[colum] / lenght
#         # return {'lenght': lenght}
    
#     def save_model(self):
#         torch.save(self.model, self.save_path)       
    
#     def _normalize_data(self, data, colums):
#         for colum in colums:
#             self.normalize_param[colum] = self._normalize(data, colum)
    
#     def _denormalize(self, tensor, mean, std) -> torch.Tensor:
#         return tensor * std
    
# trainer = Trainer(
#     data=dataset,
#     input_colums=['criteria1', 'criteria2'],
#     output_colums=['parameter1', 'parameter2'],
#     batch_size=60,
#     learning_rate=0.001,
#     random_state=30
# )

# trainer.run(15)


# type(trainer.predict(np.array([47,	3100.75])))
# print(trainer.predict(np.array([47,	3100.75])))

# trainer.save_model()