Ce projet concerne le Fine-Tuning et la manipulation d'un modèle Fine-Tuné.


Il rassemble les fonctionnalités suivantes :
* Exécuter le Fine-Tuning du modèle.
* Interroger le modèle Fine-Tuné.


La commande pour lancer ce projet est la suivante ::
uvicorn FineTuningController:app --reload --workers 1 --host 0.0.0.0 --port 8012


Commandes pour installer les dépendances :
* !pip install transformers datasets evaluate
* !pip install transformers[torch]
* !pip install accelerate==0.27.1
* !pip install accelerate -U
* !pip install transformers==4.28.0


URL DU TUTORIEL :
https://huggingface.co/docs/transformers/tasks/question_answering#inference

