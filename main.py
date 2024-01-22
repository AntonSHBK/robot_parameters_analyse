import numpy as np
from script import LoadModel, NeuralNetwork

def main():
    # # First star
    # data_1_path = Path("data/DataSet900.csv")
    # data_2_path = Path("data/DataSet1000.csv")
    # data_3_path = Path("data/DataSet100000.csv")
    # data_4_path = Path("data/DataSet129600.csv")

    # dataset = load_data(data_1_path)

    # trainer = Trainer(
    #     data=dataset,
    #     input_colums=['criteria1', 'criteria2'],
    #     output_colums=['parameter1', 'parameter2'],
    #     batch_size=60,
    #     learning_rate=0.001,
    #     random_state=30
    # )
    # trainer.run(300)
    # trainer.save_model()

    # Next starts
    load_model = LoadModel(
        'saved_model/model_weights.pth',
        'saved_model/normalize_params.pkl',
        input_colums=['criteria1', 'criteria2'],
        output_colums=['parameter1', 'parameter2']
    )
    print(load_model.predict(np.array([47,	3100.75])))

if __name__ == '__main__':
    main()