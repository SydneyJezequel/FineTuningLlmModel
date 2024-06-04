
OBJECTIF DE CE PROJET :
Ce projet concerne le Fine-Tuning et la manipulation d'un modèle Fine-Tuné.



FONCTIONNALITES :
Il rassemble les fonctionnalités suivantes :
* Exécuter le Fine-Tuning du modèle.
* Interroger le modèle Fine-Tuné.



COMMANDES POUR LANCER LE PROJET :
La commande pour lancer les controller et service en charge du Fine-Tuning :
uvicorn FineTuningController:app --reload --workers 1 --host 0.0.0.0 --port 8012

La commande pour lancer les controller et service en charge de la génération des questions et réponses :
uvicorn ExecuteModelController:app --reload --workers 1 --host 0.0.0.0 --port 8013



COMMANDES POUR INSTALLER LES DEPENDANCES :
Commandes pour installer les dépendances :
* !pip install transformers datasets evaluate
* !pip install transformers[torch]
* !pip install accelerate==0.27.1
* !pip install accelerate -U
* !pip install transformers==4.28.0



DATASET ET MODELE UTILISES :
Le dataset utilisé est : "hotpot_qa".
Le modèle utilisé est : "distilbert/distilbert-base-uncased".
Leurs configuration sont définies dans le fichier CONFIG.py.



DOSSIER "qa_model" :
Ce dossier contient le modèle à interroger.
Il faut ajouter les fichiers suivants du modèle "distilbert/distilbert-base-uncased" fine-tuné, 
-config.json
-pytorch_model.bin
-tokenizer.json



URL DU TUTORIEL :
https://huggingface.co/docs/transformers/tasks/question_answering#inference

