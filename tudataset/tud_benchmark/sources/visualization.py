from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx



def scatterPlot2DBig(data, title, classes):
    fig = plt.figure(figsize=(15, 15))
    if classes is not None:
        plt.scatter(data[:, 0], data[:, 1], c=classes)
    else:
        plt.scatter(data[:, 0], data[:, 1])
    plt.title(title,fontsize=18)
    plt.show()

def scatterPlot2DMiddle(data, title, classes):
    fig = plt.figure(figsize=(15, 15))
    if classes is not None:
        plt.scatter(data[:, 0], data[:, 1], c=classes, s=0.2)
    else:
        plt.scatter(data[:, 0], data[:, 1], s=0.2)
    plt.title(title,fontsize=18)
    plt.show()

def showBoxplot(data, title):
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(data=data[:, 0:2], columns=(0, 1))
    df = df.rename(columns={0: 'First principal component', 1: 'Second principal component'})
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=18)
    ax1 = fig.add_subplot(221)
    ax1 = sns.boxplot(x=df['First principal component'])
    ax2 = fig.add_subplot(222)
    ax2 = sns.boxplot(y=df['Second principal component'])
    plt.show()

def visualize(G, color=None, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=color,
                     cmap="Set2")
    plt.show();

def showEntireDataset(wl_listG, wl_listV, tsvd_graphlet_vectors, kpca_graphlet_gram, tsvd_shortestpath_vectors, kpca_shortestpath_gram, classes):
    for i in range(1, 8):
        if (i == 6):
            data_tsvd = tsvd_graphlet_vectors
            data_kpca = kpca_graphlet_gram
        elif (i == 7):
            data_tsvd = tsvd_shortestpath_vectors
            data_kpca = kpca_shortestpath_gram
        else:
            data_tsvd = wl_listV[i - 1]
            data_kpca = wl_listG[i - 1]
        fig = plt.figure(figsize=(15, 15))
        if (i == 6):
            fig.suptitle('Graphlet', fontsize=25)
        elif (i == 7):
            fig.suptitle('Shortest Path', fontsize=25)
        else:
            fig.suptitle(f'Weisfeiler-Lehman {i}', fontsize=25)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.title.set_text('2D TruncatedSVD')
        ax2.title.set_text('2D KernelPCA')
        ax3.title.set_text('3D TruncatedSVD')
        ax4.title.set_text('3D KernelPCA')
        ax1.scatter(data_tsvd[:, 0], data_tsvd[:, 1], c=classes)
        ax2.scatter(data_kpca[:, 0], data_kpca[:, 1], c=classes)
        ax3.scatter3D(data_tsvd[:, 0], data_tsvd[:, 1], data_tsvd[:, 2], c=classes)
        ax4.scatter3D(data_kpca[:, 0], data_kpca[:, 1], data_kpca[:, 2], c=classes)
        plt.show()
        print("________________________________________________________________________________________")
        print()
