# Project Racoon

This repository contains my work, and instructions to reproduce it, for the PClub Secretary Recruitment Task - Project Racoon. This implements a very basic Federated Learning algorithm using PyTorch and Scikit-Learn without using any frameworks/libraries dedicated to FL algorithms. The `datasets` folder contains the split data for each of the datasets into 3 parts.

I have chosen to work with a local client instead of a web-based implementation of local training to minimize the work needed, since it is supposed to be a PoC. Pre-processing has been included in the Local Training Client, given the expectation that data uploaded by various clients will not be pre-processed. 

### Deployment
The project may be deployed in a "single" command by pasting the following into your shell (assuming a Linux Desktop Environment), while you are in your Projects directory (where you would want the folder Racoon to be present in). You may change the folder name where the application is initialized by changing the only argument of mkdir. The command is just a bunch of commands thrown together for a good experience. 
```sh
mkdir Racoon && cd $_ && git clone https://github.com/sjais1337/racoon . && python3 -m venv venv && source venv/bin/activate && pip install django djangorestframework djangorestframework-simplejwt django-cors-headers torch numpy pandas scikit-learn && clear && python manage.py makemigrations fl_platform && python manage.py makemigrations accounts && python manage.py migrate && clear && python manage.py initialize_groups && python manage.py createsuperuser
```
It creates a directory Racoon, cds into it, clones the repo in the directory Racoon, activates the virtual environment, installs the required python packages, makes the migrations, initializes the training groups and finally prompts the user for creating a super user account. This super user account is necessary, since it'll be the one which will initialize the training, and move the training to the next rounds for all the other users. 

If there are any issues while running the commands above, you can just run the following commands one by one,
```sh
mkdir Racoon && cd $_
git clone https://github.com/sjais1337/racoon .
python3 -m venv venv
source venv/bin/activate
pip install django djangorestframework djangorestframework-simplejwt django-cors-headers torch numpy pandas scikit-learn
python manage.py makemigrations fl_platform
python manage.py makemigrations accounts
python manage.py migrate
python manage.py initialize_groups
python manage.py createsuperuser
```
The `python manage.py initialize_groups` is a django admin command to initialize the 4 training groups, and load them into the database. 
