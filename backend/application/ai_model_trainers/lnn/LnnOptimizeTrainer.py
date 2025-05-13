import itertools
import logging

from application.ai_model_trainers.lnn.LnnTrainer import LNN_Trainer

# Настройка модуля логирования
logging.basicConfig(
    filename='app.log',      # Имя файла журнала
    filemode='a',            # Открытие файла для добавления записей ('w' — перезапись)
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_message(message):
    logging.info(message)

class LnnOptimizeTrainer:
    def __init__(
            self,
            ai_model,
            epochs,
            hidden_layers,
            batch_sizes,
            neurons_per_layers,
            activation_functions,
            optimizers,
            user_name,
            dataset_name):
        self.ai_model = ai_model

        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.batch_sizes = batch_sizes
        self.neurons_per_layer = neurons_per_layers
        self.activation_functions = activation_functions
        self.optimizers = optimizers

        self.user_name = user_name
        self.dataset_name = dataset_name

        self.total_count = 0

    def optimize_train(self, trained_model_name, is_create_app):
        self.test_loss = 1
        self.test_acc = 0
        num_layers = 0

        try:
            # Итерация по числу скрытых слоёв
            for layers in self.hidden_layers:
                log_message(f'Layers {num_layers}/{len(self.hidden_layers)}')
                # Генерируем комбинации нейронов
                neuron_combinations = list(itertools.product(self.neurons_per_layer, repeat=layers))

                # Генерируем комбинации функций активаций
                activation_combinations = list(itertools.product(self.activation_functions, repeat=layers))

                # Полностью совмещаем нейроны и функции активаций между собой
                full_combinations = [
                    {'neurons': neurons, 'activations': activations}
                    for neurons in neuron_combinations
                    for activations in activation_combinations
                ]

                num_combo_neurons_activations = 0
                for combo in full_combinations:
                    log_message(f'Combo neurons and activations {num_combo_neurons_activations}/{len(full_combinations)}')
                    self.__iteration_for_optimizers(list(combo['neurons']), list(combo['activations']))
                    num_combo_neurons_activations += 1
                num_layers += 1
                
        except Exception as ex:
            log_message(ex)
        finally:
            log_message(f'Total count {self.total_count}')
            return self.trainer.save_model(trained_model_name, self.best_trained_model, self.history, is_create_app)

    def __iteration_for_optimizers(self, neurons_in_layers, activations):
        for optimizer in self.optimizers:
            self.__iteration_for_batch_size(neurons_in_layers, activations, optimizer)

    def __iteration_for_batch_size(self, neurons_in_layers, activations, optimizer):
        for batch_size in self.batch_sizes:
            self.__iteration_for_epochs(neurons_in_layers, activations, optimizer, batch_size)


    def __iteration_for_epochs(self, neurons_in_layers, activations, optimizer, batch_size):
        for epoch in self.epochs:
            trainer = LNN_Trainer(
                self.ai_model,
                epoch,
                batch_size,
                neurons_in_layers,
                activations,
                optimizer,
                self.user_name,
                self.dataset_name,
                50,
                25)
            trained_model, history, test_loss, test_acc = trainer.train()

            if (self.test_loss > test_loss and self.test_acc < test_acc):
                self.trainer = trainer
                self.test_loss = test_loss
                self.test_acc = test_acc
                self.best_trained_model = trained_model
                self.history = history
            
            self.total_count += 1
            log_message(f'Values acc - {test_acc} Values loss - {test_loss}; Configuration: epoch - {epoch}; batch_size - {batch_size}; neurons_in_layers - {neurons_in_layers}; activations - {activations}; optimizer - {optimizer}')
