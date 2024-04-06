# Параметры для запуска скриптов

    Файл со значениями параметров, необходимых к вводу для запуска скриптов не в IDE, а на удаленной машине

## 1) cnn_cv_regression_2layers.py \

    Скрипт для обучения сверточной нейронной сети с 2 слоями свертки. Параметры конфигурации скрипта:

    path_tuner_log = "cnn_tuner_3_layers"
    path_learn_log = "cnn_learn_3_layers"
    path_df = '../datasets/total_df_for_aio.csv'
    path_images = '../vigna_AIO_set'
    model_save_folder = 'model_saves'
    idx_save_folder = '../idx_files'
    accuracy_save_folder = '../accuracies'

## 2) cnn_cv_regression_3layers.py \

    Скрипт для обучения сверточной нейронной сети с 3 слоями свертки. Параметры конфигурации скрипта:

    path_tuner_log = "cnn_tuner_3_layers"
    path_learn_log = "cnn_learn_3_layers"
    path_df = '../datasets/total_df_for_aio.csv'
    path_images = '../vigna_AIO_set'
    model_save_folder = 'model_saves'
    idx_save_folder = '../idx_files'
    accuracy_save_folder = '../accuracies'

## 3) regression_test.py \

    Скрипт, дублирующий ноутбук Дениса с обучением регрессионной модели на базе features, получаемых при помощи
    обученного словаря и нейронной сети.

## 4) custom_visualization.py \

    Скрипт, содержащий кастомные функции, используемые для визуализации результатов обучения/валидации/тестирования
    модели.

## 5) graph_cnn_results.py \

    Общий скрипт, выполнящий визуализацию результатов обучения/валидации/тестирования модели и подбора гиперпараметров.
    Параметры конфигурации:

    path_to_model = "tmp_scripts_for_models/models_for_demonstration/13_08_23_7_layers/7_regr_model_trained_cross_2023-08-13.h5"
    path_to_tuner = 'tmp_scripts_for_models/models_for_demonstration/13_08_23_7_layers/tuner_2layers'
    path_to_df = "../tables_for_AIO/total_df_for_aio.csv"
    path_to_indices = "tmp_scripts_for_models/models_for_demonstration/13_08_23_7_layers/7_train_idx_2023-08-08.txt"
    path_to_accuracies = "tmp_scripts_for_models/models_for_demonstration/13_08_23_7_layers/accuracies"
    path_to_images = "../AIO_set/vigna/for_model"

## 6) more_cnn.py \

    Скрипт, в котором потенциально потестирую возможность применения других сверточных нейронных сетей.

## 7) cnn_architecture_variants.py \

    Неисполняемый скрипт, содержащий ранние варианты архитектуры сверточной нейронной сети с 2/3 сверточными слоями.
    Для быстрого back-up, чтобы гит не откатывать.