# Description
This project aims to realize a MLP architecture to solve a triple classification task

# Requirements
torch=1.11.0+cu113

# Quick start
you can visualize the final results by
```pwd
python main.py
```

# Component
- main.py: main program for training, testing and visualisation, you must change the parameters inside this program
- main.ipynb: the notebook version of main.py
- scripts: directory which contains different scripts
    - animator.py: contains necessary class of animator for display the processes of training
    - visualisation.py: visualise the performance of model
    - test_initialisation.py: test different methods of parameters initialisation
    - test_optimizer.py: test different optimizer
    - test_sgd.py: test different learning rate
    - test_batch_size.py: test different batch size
    - datasets.py: class of E3datasets
    - model.py: class of E3model
    - trainer.py: functions for training
- models: contains model parameters pth
    - base.pth: after 5000 epochs of training
