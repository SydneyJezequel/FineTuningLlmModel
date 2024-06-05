from BO.DatasetManager import DatasetManager
from BO.Model import Model
from BO.TokenizerManager import TokenizerManager
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator
import config






class FineTuningService:
    """ Service en charge du Fine Tuning """



    def __init__(self):
        """ Constructeur """
        self.datasetManager = DatasetManager(config.DATASET_NAME, config.SUB_DATASET_NAME)
        self.tokenizerManager = TokenizerManager(config.MODEL_NAME)
        self.model = Model(config.MODEL_NAME)



    def fine_tune_model(self, num_epochs=config.NUM_EPOCH, train_dataset_size=config.TRAIN_DATASET_SIZE, validation_dataset_size=config.VALIDATION_DATASET_SIZE,
                        train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE, eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE):
        """ Méthode en charge du Fine Tuning """
        # Chargement des dataset :
        print("début du Fine Tuning")
        train_dataset = self.datasetManager.prepare_dataset(self.datasetManager.dataset, train_dataset_size, config.TRAIN_TYPE)
        validation_dataset = self.datasetManager.prepare_dataset(self.datasetManager.dataset, validation_dataset_size, config.VALIDATION_TYPE)
        print("train dataset : ", train_dataset)
        print("validation dataset : ", validation_dataset)
        # Tokenization des datasets :
        tokenized_train_dataset = train_dataset.map(
            lambda examples: self.datasetManager.preprocess_function(examples, self.tokenizerManager.tokenizer), batched=True)
        tokenized_validation_dataset = validation_dataset.map(
            lambda examples: self.datasetManager.preprocess_function(examples, self.tokenizerManager.tokenizer), batched=True)
        # Création du batch :
        data_collator = DefaultDataCollator()
        # Définition des Hyperparamètres :
        self.training_arguments = TrainingArguments(
            output_dir=config.MODEL_PATH,
            evaluation_strategy=config.EVALUATION_STRATEGY,
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=config.WEIGHT_DECAY)
        # Exécution du Fine Tuning :
        self.fine_tuning_execution(self.model, self.training_arguments, tokenized_train_dataset,
                                   tokenized_validation_dataset, self.tokenizerManager, data_collator)



    def fine_tuning_execution(self, model, training_arguments, tokenized_train_dataset, tokenized_validation_dataset, tokenizer, data_collator):
        """ Méthode en charge du Fine-Tuning du modèle """
        # Configuration du Trainer :
        trainer = Trainer(
            model=model.model,
            args=training_arguments,        # Hyperparamètres.
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,    # batch/lot pour traiter les données.
        )
        # Exécution de l'entrainement :
        trainer.train()

