# Generate turn-based and action RPG stats

The main purpose of this program is to train and measure the suitability of the turn-based RPG stats or gameplay attributes using neural network algorithm. You can looks at `datasets` directory to see the input example of this program. This project contains three main programs, the first is the example that I took is `dota_2_multiclass_classification`, it uses to classify all the Dota 2 characters accorting to their types. The second and the thrid was the intepretation on the first project, it uses to classify all the roles of the player characters and the type of the enemy squentially.

## Prerequisite

Here are the packages that need to install:

+ matplotlib==3.3.3
+ numpy==1.19.3
+ pandas==1.1.5
+ keras==2.3.1
+ tensorflow==1.15

This comment below uses to install all the required package of this program.

        $ pip install -r requirement.txt

## How to run in Linux or Windows

This comment uses to classify the Dota 2 stats or gameplay attributes:

        $ python dota_2_multiclass_classification.py

This comment uses to classify the role of the player by its stats or gameplay attributes:

        $ python player_multiclass_classification.py

This comment uses to classify the type of the enemy by its stats or gameplay attributes:

        $ python enemy_multiclass_classification.py

## Program Content Explanation

This program contain these files and folders:

+ `dota_2_multiclass_classification.py`: The program to classify the Dota 2 stats or gamelay attribute with neural network multiclass-classification.
+ `player_multiclass_classification.py`: The program to classify the Player characters roles in RPG games by its stats or gamelay attribute with neural network multiclass-classification.
+ `enemy_multiclass_classification.py`: The program to classify the Enemy characters types in RPG games by its stats or gamelay attribute with neural network multiclass-classification.
+ `datasets`: The input of the program, all the Dota 2 characters, the player and the enemy stats will represent with `.CSV` here.
+ `tensorboard`: This is the visualization of the training process, all the graphs represents the result of the player and the enemy stat growth.
