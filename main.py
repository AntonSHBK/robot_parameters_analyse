import numpy as np
from pathlib import Path
# from script import LoadModel, NeuralNetwork
from scripts.simple_analise import LoadModel, NeuralNetwork, load_data, Trainer

def main():
    # First star
    data_1_path = Path("data/DataSet900.csv")
    data_2_path = Path("data/DataSet1000.csv")
    data_3_path = Path("data/DataSet100000.csv")
    data_4_path = Path("data/DataSet129600.csv")

    dataset = load_data(data_1_path)
   
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
    trainer_1.plot_train_result()

    trainer_1.save_model()

    # Next starts
    load_model = LoadModel(
        'saved_model/criteria1_criteria2_parameter1_parameter2_model_weights.pth',
        'saved_model/criteria1_criteria2_parameter1_parameter2_normalize_params.pkl',
        input_colums=['criteria1', 'criteria2'],
        output_colums=['parameter1', 'parameter2']
    )
    print(load_model.predict(np.array([119,	25795])))
    print(dataset.iloc[178:180])

if __name__ == '__main__':
    main()