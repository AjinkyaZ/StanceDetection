import matplotlib.pyplot as plt



def graph_losses(losses, model_name):
    N = len(losses)
    if N==0: return
    x = [i for i in range(N)]
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.title('Loss over time for %s'%model_name)
    plt.plot(x, losses, label=model_name)
