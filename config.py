# Paramètres de la classe Model :
MODEL_NAME = "distilbert/distilbert-base-uncased"



# Paramètres de la classe Dataset :
DATASET_NAME = "hotpot_qa"
SUB_DATASET_NAME = "distractor"
SEED_VALUE = 42
TRAIN_DATASET_SIZE = 8000
VALIDATION_DATASET_SIZE = 2000
TRAIN_TYPE = "train"
VALIDATION_TYPE = "validation"



# Paramètres des Services FineTuning et ExecuteModel :
MODEL_PATH = "qa_model"                 # Répertoire de sortie du fichier de poids du modèle.
EVALUATION_STRATEGY = 'epoch'           # L’évaluation a lieu à la fin de chaque Epoch.
LEARNING_RATE = 2e-5                    # Taux d’ajustement des poids du modèle.
PER_DEVICE_TRAIN_BATCH_SIZE = 16        # Nombre de lignes d’entrainement par batch.
PER_DEVICE_EVAL_BATCH_SIZE = 16         # Nombre de lignes d’évaluation par batch.
NUM_EPOCH = 3                           # Nombre d’Epoch (cycle complet d’entrainement).
WEIGHT_DECAY = 0.01                     # Pénalise les poids du modèle plus grands, contribuant ainsi à prévenir le surapprentissage en limitant la complexité du modèle.
TENSOR_TYPE = "pt"                      # Type de Tenseur.
ACTIVITY_TYPE = "question-answering"

