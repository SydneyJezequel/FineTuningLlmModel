from pydantic import BaseModel






class FineTuningParameters(BaseModel):
    """ Paramètres attendus par la méthode fine_tuning_service::fine_tune_model() """



    num_epochs: int
    train_dataset_size: int
    validation_dataset_size: int
    train_batch_size: int
    eval_batch_size: int


