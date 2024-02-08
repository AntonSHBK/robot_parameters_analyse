# %% [markdown]
# # Импорты

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %% [markdown]
# ## Дефолтные настройки Matplotlib

# %%
import matplotlib.pyplot as plt

# устанавливаем дефолтные размеры шрифтов
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# %%
convert_type_dict = {'parameter1': float,
                    'parameter2': float,
                    'criteria1': float,
                    'criteria2': float,
                    'constraint1': bool,
                    'constraint2': bool,
                    }

# %% [markdown]
# Загрузка датасета

# %%
def load_data(path: Path):
    print("Read data set from path {path}".format(path=path))
    df = pd.read_csv(path).astype(convert_type_dict)
    return df

# %% [markdown]
# Непосредственно загрузка

# %%
data_1_path = Path("data/DataSet900.csv")
data_2_path = Path("data/DataSet1000.csv")
data_3_path = Path("data/DataSet100000.csv")
data_4_path = Path("data/DataSet129600.csv")

data_1 = load_data(data_1_path)
data_2 = load_data(data_2_path)
data_3 = load_data(data_3_path)
data_4 = load_data(data_4_path)

dataset = data_1

# %% [markdown]
# Шапка датасета, первые 5 записей  набора данных.

# %% [markdown]
# Код описывающий сохранение графиков и рисунков

# %%
IMAGES_PATH = Path() / "imgs"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %% [markdown]
# Устанавливаем параметр рандомизации (что бы значения повторялись при запуске)

# %%
np.random.seed(30)

# %% [markdown]
# Новый критерий сильно коррелирует с нашим параметром

# %%
import torch
from typing import List
import pickle
from torch import nn
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## Установить роботу с куда или нна цп

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %% [markdown]
# # Working with data

# %%
class MyDataset(Dataset):
    '''
    Dataset:
    '''
    def __init__(self, 
                 data:pd.DataFrame,
                 input_colums: List[str],
                 output_colums: List[str]) -> None:
        self.input_colums = input_colums
        self.output_colums = output_colums
        
        x = data[self.input_colums].values
        y = data[self.output_colums].values
        
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)      
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]    

# %% [markdown]
# # Creating Models

# %%

class NeuralNetwork(nn.Module):
    '''
    Define neural network model:
    '''
    def __init__(self, inputs, outputs, middle_layers=128):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(            
            nn.Linear(inputs, middle_layers),
            nn.ReLU(),
            nn.Linear(middle_layers, middle_layers),
            nn.ReLU(),
            nn.Linear(middle_layers, middle_layers),
            nn.ReLU(),
            nn.Linear(middle_layers, outputs)
        )

    def forward(self, x: torch.Tensor):
        logits = self.linear_relu_stack(x)
        return logits

test = NeuralNetwork(2,2).to(device)
print(test)

# %% [markdown]
# 

# %% [markdown]
# # Optimizing the Model Parameters

# %%
from numpy import mean
from sympy import prime
from torch.nn import MSELoss
from torch.optim import SGD

from sklearn.model_selection import train_test_split


class Trainer():
    '''
    Trainer:
    '''
    def __init__(self,
                 data:pd.DataFrame,
                 input_colums:List[str],
                 output_colums:List[str],
                 batch_size=32,
                 learning_rate=0.005,
                 shuffle=False,
                 test_size=0.2,
                 random_state=None,
                 save_path='saved_model/',
                 model_middle_layers=512):
        
        self.save_path = save_path
        
        self.input_colums = input_colums
        self.output_colums = output_colums
        
        self.uniq_name = '_'.join(input_colums + output_colums)
        
        self.data = data.copy()
        self.normalize_param = self._normalize_data(self.data, input_colums+output_colums)
        with open(save_path+self.uniq_name+'_normalize_params.pkl', 'wb') as file: 
            pickle.dump(self.normalize_param, file)

        train_set, test_set = train_test_split(self.data, test_size=test_size, random_state=random_state)
                
        self.train_dataloader = DataLoader(
            MyDataset(train_set, input_colums, output_colums),
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            MyDataset(test_set, input_colums, output_colums),
            batch_size=batch_size,
            shuffle=False
        )
        
        self.model = NeuralNetwork(
            inputs=len(input_colums),
            outputs=len(output_colums),
            middle_layers=model_middle_layers
        ).to(device)
        
        self.loss_func = MSELoss()
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)  
        
        self.validate_loss_list = []     
        self.train_loss_list = []
    
    def train(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        train_loss = 0.
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss: torch.Tensor = self.loss_func(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
        
        avg_loss = train_loss/len(self.train_dataloader)
        self.train_loss_list.append(avg_loss)
        return avg_loss
                
    def validate(self):
        self.model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                val_loss_sum += self.loss_func(pred, y).item()
        avg_loss = val_loss_sum/len(self.test_dataloader)
        self.validate_loss_list.append(avg_loss)
        return avg_loss        
        
    def run(self,  epochs=10):
        for t in range(epochs):
            train_avg_loss = self.train()
            
            val_avg_loss = self.validate()
            counter_1 = epochs / 5
            if t % counter_1 == 0:
                print(f'Epoch [{t + 1:03}/{epochs:03}] | Train Loss: {train_avg_loss:.6f}')
                print(f"Validation AVG Loss: {val_avg_loss:>12f} \n")
            
        print("Done!")
    
    # def predict(self, input: np.array):
    #     print('Входные данные:', input)
    #     for index, item in enumerate(self.input_colums):
    #         input[index] = (input[index] - self.normalize_param[item]['mean']) /\
    #             self.normalize_param[item]['std']
    #     print('Нормализованные:', input)
    #     pred:torch.Tensor = self.model(torch.tensor(input, dtype=torch.float32))        
    #     pred = pred.detach().numpy()
    #     print('Вывод модели:', pred)
    #     for index, item in enumerate(self.output_colums):
    #         # pred[index] = pred[index] * self.normalize_param[item]['lenght']
    #         pred[index] = pred[index] * self.normalize_param[item]['std'] +\
    #             self.normalize_param[item]['mean']
    #     print('Денормализованные данные:', pred)
    #     return pred  
    
    def predict(self, input: np.array):
        input = input.astype(float)
        print('Входные данные:', input)
        for index, item in enumerate(self.input_colums):
            input[index] = 2 * ((input[index] - self.normalize_param[item]['min']) /\
                (self.normalize_param[item]['max'] - self.normalize_param[item]['min'])) -1
        print('Нормализованные:', input)
        pred:torch.Tensor = self.model(torch.tensor(input, dtype=torch.float32))        
        pred = pred.detach().numpy()
        print('Вывод модели:', pred)
        for index, item in enumerate(self.output_colums):
            pred[index] = (1 + pred[index]) / 2 * \
                (self.normalize_param[item]['max'] - self.normalize_param[item]['min']) +\
                    self.normalize_param[item]['min']
        print('Денормализованные данные:', pred)
        return pred  
    
    def _normalize(self,
                   data: pd.DataFrame,
                   colum: str) -> torch.Tensor:
        max_val = data[colum].max()
        min_val = data[colum].min()
        data[colum] = 2 * ((data[colum] - min_val) / (max_val - min_val)) -1
        return {'max': max_val, 'min': min_val}
    
    # def _normalize(self,
    #                data: pd.DataFrame,
    #                colum: str) -> torch.Tensor:
    #     mean = data[colum].mean()
    #     std = data[colum].std()
    #     data[colum] = (data[colum] - mean) / std  
    #     return {'mean': mean, 'std': std}
    
    def save_model(self):
        torch.save(self.model, self.save_path+self.uniq_name+'_model_weights.pth')       
    
    def _normalize_data(self, data, colums):
        params = {}
        for colum in colums:
            params[colum] = self._normalize(data, colum)
        return params
    
    def _denormalize(self, tensor, mean, std) -> torch.Tensor:
        return tensor * std
    
    def plot_train_result(self):
        plt.plot(range(len(self.train_loss_list)), self.train_loss_list, self.validate_loss_list)

# %%
class LoadModel():
    def __init__(self, model_path, normalize_params_path, input_colums, output_colums) -> None:
        self.input_colums = input_colums
        self.output_colums = output_colums
        self._model_path = model_path
        with open(normalize_params_path, 'rb') as file: 
            self.normalize_param = pickle.load(file)
        self.model = torch.load(model_path)
        self.model.eval()
    
    def predict(self, input):
        input = input.astype(float)
        print('Входные данные:', input)
        for index, item in enumerate(self.input_colums):
            input[index] = 2 * ((input[index] - self.normalize_param[item]['min']) /\
                (self.normalize_param[item]['max'] - self.normalize_param[item]['min'])) -1
        print('Нормализованные:', input)
        pred:torch.Tensor = self.model(torch.tensor(input, dtype=torch.float32))        
        pred = pred.detach().numpy()
        print('Вывод модели:', pred)
        for index, item in enumerate(self.output_colums):
            pred[index] = (1 + pred[index]) / 2 * \
                (self.normalize_param[item]['max'] - self.normalize_param[item]['min']) +\
                    self.normalize_param[item]['min']
        print('Денормализованные данные:', pred)
        return pred

if __name__ == '__main__':
    # %% [markdown]
    # # Анализ

    # %%
    # First star
    data_1_path = Path("data/DataSet900.csv")
    data_2_path = Path("data/DataSet1000.csv")
    data_3_path = Path("data/DataSet100000.csv")
    data_4_path = Path("data/DataSet129600.csv")

    dataset = load_data(data_1_path)

    # %%
    dataset.head()

    # %% [markdown]
    # # Проверка 1 
    # input_colums=['criteria1', 'criteria2'],
    # 
    # output_colums=['parameter1', 'parameter2'],

    # %%
    trainer_1 = Trainer(
        data=dataset,
        input_colums=['criteria1', 'criteria2'],
        output_colums=['parameter1', 'parameter2'],
        batch_size=1000,
        learning_rate=0.1,
        shuffle=False,
        test_size=0.2,
        random_state=30,
        save_path='saved_model/',
        model_middle_layers=1024,
    )
    trainer_1.run(1000)


    # %%
    trainer_1.plot_train_result()

    # %%
    trainer_1.save_model()
    trainer_1.predict(np.array([119, 25795]))

    print(trainer_1.normalize_param)

    # Next starts
    load_model = LoadModel(
        'saved_model/criteria1_criteria2_parameter1_parameter2_model_weights.pth',
        'saved_model/criteria1_criteria2_parameter1_parameter2_normalize_params.pkl',
        input_colums=['criteria1', 'criteria2'],
        output_colums=['parameter1', 'parameter2']
    )
    print(load_model.predict(np.array([119,	25795])))

    # %%
    dataset.iloc[178:180]

    # %% [markdown]
    # # Проверка 2
    # input_colums=['parameter1', 'parameter2'],
    # 
    # output_colums=['criteria1', 'criteria2'],

    # %%
    trainer_2 = Trainer(
        data=dataset,
        input_colums=['parameter1', 'parameter2'],
        output_colums=['criteria1', 'criteria2'],
        batch_size=1000,
        learning_rate=0.5,
        shuffle=False,
        test_size=0.2,
        random_state=30,
        save_path='saved_model/',
        model_middle_layers=1024,
    )
    trainer_2.run(1000)


    # %%
    trainer_2.plot_train_result()

    # %%
    trainer_2.save_model()
    trainer_2.predict(np.array([25., 94.]))

    print(trainer_2.normalize_param)

    # %%
    dataset.iloc[178:180]

    # %% [markdown]
    # # Проверка 3
    # input_colums=['parameter1', 'criteria1'],
    # 
    # output_colums=['parameter2', 'criteria2'],

    # %%
    trainer_3 = Trainer(
        data=dataset,
        input_colums=['parameter1', 'criteria1'],
        output_colums=['parameter2', 'criteria2'],
        batch_size=1000,
        learning_rate=0.1,
        shuffle=False,
        test_size=0.2,
        random_state=30,
        save_path='saved_model/',
        model_middle_layers=1024,
    )
    trainer_3.run(1000)


    # %%
    trainer_3.plot_train_result()

    # %%
    trainer_3.save_model()
    trainer_3.predict(np.array([25., 119.]))

    print(trainer_3.normalize_param)

    # %%
    dataset.iloc[178:180]

    # %% [markdown]
    # # Проверка 4
    # input_colums=['parameter2', 'criteria2'],
    # 
    # output_colums=['parameter1', 'criteria1'],

    # %%
    trainer_4 = Trainer(
        data=dataset,
        input_colums=['parameter2', 'criteria2'],
        output_colums=['parameter1', 'criteria1'],
        batch_size=1000,
        learning_rate=0.1,
        shuffle=False,
        test_size=0.2,
        random_state=30,
        save_path='saved_model/',
        model_middle_layers=1024,
    )
    trainer_4.run(1000)


    # %%
    trainer_4.plot_train_result()

    # %%
    trainer_4.save_model()
    trainer_4.predict(np.array([94., 25795.]))

    print(trainer_4.normalize_param)

    # %%
    dataset.iloc[178:180]


