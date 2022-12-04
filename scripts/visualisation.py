
def visualisation_results(test_dataloader, model):
    
    for X,y in test_dataloader:
        pred = model(X)
