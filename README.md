# Разработка и анализ нейронной сети для обработки данных кинематики роботов

![Фон](https://github.com/AntonSHBK/simple_robot_parameters_analyse/blob/main/article_imgs/face.png?raw=true)

## [Основной анализ (Jupyter Notebook)](analise.ipynb)

## [Статья с описание](docs/article.md)

## Описание
Проект посвящен разработке и тестированию модели машинного обучения для анализа данных кинематики простых роботов. Используя нейронные сети в библиотеке PyTorch, модель обучается предсказывать поведение робота на основе предоставленных данных.

## Как начать
### Установка зависимостей
Для установки необходимых библиотек выполните следующую команду:
```
pip install -r requirements.txt
```

### Анализ данных

[Исходный датасет](https://github.com/AntonSHBK/simple_robot_parameters_analyse/blob/main/data/DataSet900.csv)

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parameter1</th>
      <th>parameter2</th>
      <th>criteria1</th>
      <th>criteria2</th>
      <th>constraint1</th>
      <th>constraint2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>20</td>
      <td>0.000</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>13</td>
      <td>23</td>
      <td>216.770</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>16</td>
      <td>26</td>
      <td>490.088</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>19</td>
      <td>29</td>
      <td>819.956</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>22</td>
      <td>32</td>
      <td>1206.370</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

Для анализа данных используйте Jupyter notebooks (`analise.ipynb` и `analise_data.ipynb`), которые включают в себя визуализации и предварительный анализ.

## Анализ результатов

В качестве входных  параметров модели были выбраны: `parameter1`, `parameter2`, `constraint1`, `constraint2`. качестве выходных параметров были выбраны: `criteria1`, `criteria2`.

![Анализ обучения](https://github.com/AntonSHBK/simple_robot_parameters_analyse/blob/main/article_imgs/MAE_metrics.png?raw=true)

```txt
Epoch 10, Loss: 0.2909, MAE: 0.4163, Val Loss: 0.2987, Val MAE: 0.4313
Epoch 20, Loss: 0.0782, MAE: 0.2165, Val Loss: 0.0852, Val MAE: 0.2287
Epoch 30, Loss: 0.0275, MAE: 0.1166, Val Loss: 0.0326, Val MAE: 0.1259
Epoch 40, Loss: 0.0143, MAE: 0.0862, Val Loss: 0.0168, Val MAE: 0.0893
Epoch 50, Loss: 0.0083, MAE: 0.0653, Val Loss: 0.0098, Val MAE: 0.0685
Epoch 60, Loss: 0.0052, MAE: 0.0519, Val Loss: 0.0063, Val MAE: 0.0544
Epoch 70, Loss: 0.0035, MAE: 0.0423, Val Loss: 0.0045, Val MAE: 0.0463
Epoch 80, Loss: 0.0026, MAE: 0.0371, Val Loss: 0.0035, Val MAE: 0.0414
Epoch 90, Loss: 0.0020, MAE: 0.0330, Val Loss: 0.0028, Val MAE: 0.0376
Epoch 100, Loss: 0.0016, MAE: 0.0298, Val Loss: 0.0023, Val MAE: 0.0348

```

## Документация
Дополнительная документация и описания методологии доступны в каталоге `docs`.