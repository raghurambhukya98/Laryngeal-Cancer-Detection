from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

No_of_Dataset = 1


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])

def Plot_ROC_Curve():
    cls = ['DCNN', 'CNN', 'DNN', 'EnClassNet', 'EAW-HSOA-EnClassNet']
    for a in range(No_of_Dataset):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset - ' + str(a + 1) + ' - ROC Curve')
        colors = ["blue", "darkorange", "limegreen", "deeppink", "black"]
        Y_Score = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)
        for i, color in enumerate(colors):  # For all classifiers
            Predicted = Y_Score[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()

def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-EnClassNet', 'AZO-EnClassNet', 'CWO-EnClassNet', 'HSOA-EnClassNet', 'EAW-HSOA-EnClassNet']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure(facecolor='#f0f0f0')
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='AOA-EnClassNet')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='AZO-EnClassNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='CWO-EnClassNet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='HSOA-EnClassNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EAW-HSOA-EnClassNet')
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()





def Line_PlotTesults(): # algorithm
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Dataset')
            ax.yaxis.grid()
            X = np.arange(5)
            bars1 = plt.bar(X + 0.00, Graph[:, 0], color='#0f4c5c', width=0.18, label="AOA-EnClassNet")
            bars2 = plt.bar(X + 0.18, Graph[:, 1], color='#fed9b7', width=0.18, label="AZO-EnClassNet")
            bars3 = plt.bar(X + 0.36, Graph[:, 2], color='#2ec4b6', width=0.18, label="CWO-EnClassNet")
            bars4 = plt.bar(X + 0.54, Graph[:, 3], color='#ffd60a', width=0.18, label="HSOA-EnClassNet")
            bars5 = plt.bar(X + 0.72, Graph[:, 4], color='#000814', width=0.18, label="EAW-HSOA-EnClassNet")

            for bars in [bars1, bars2, bars3, bars4, bars5]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.9, f'{int(height):.0f}',
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='None', edgecolor='k', alpha=0.8))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.30, ('100', '200', '300', '400', '500'),
                       fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('Hidden Neuron Count', fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plots_Results(): # method
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Dataset')
            ax.yaxis.grid()
            X = np.arange(5)
            bars1 = plt.bar(X + 0.00, Graph[:, 5], color='#335c67', width=0.18, label="DCNN")
            bars2 = plt.bar(X + 0.18, Graph[:, 6], color='#d0f4de', width=0.18, label="CNN")
            bars3 = plt.bar(X + 0.36, Graph[:, 7], color='#e09f3e', width=0.18, label="DNN")
            bars4 = plt.bar(X + 0.54, Graph[:, 8], color='#4cc9f0', width=0.18, label="EnClassNet")
            bars5 = plt.bar(X + 0.72, Graph[:, 4], color='#540b0e', width=0.18, label="EAW-HSOA-EnClassNet")

            for bars in [bars1, bars2, bars3, bars4, bars5]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02 * height, f'{int(height):.0f}',
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='None', edgecolor='k', alpha=0.8))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.30, ('100', '200', '300', '400', '500'),
                       fontname="Arial", fontsize=1, fontweight='bold', color='k')
            plt.xlabel('Hidden Neuron Count', fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
            path = "./Results/%s_mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Table():
    eval = np.load('Eval.npy', allow_pickle=True)
    Algorithm = ['Kfold', 'AOA-EnClassNet', 'AZO-EnClassNet', 'CWO-EnClassNet', 'HSOA-EnClassNet', 'EAW-HSOA-EnClassNet']
    Classifier = ['Kfold', 'DCNN', 'CNN', 'DNN', 'EnClassNet', 'EAW-HSOA-EnClassNet']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 8]).astype(int)
    Table_Terms = [0, 2, 8]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = ['Kfold 1', 'Kfold 2', 'Kfold 3', 'Kfold 4', 'Kfold 5']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Kfold)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Confusion():
    no_of_Dataset = 1
    for n in range(no_of_Dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)
        class_2 = ['HBV', 'HE', 'IPCL', 'LE']
        confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_2)
        cm_display.plot()
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    # Plot_ROC_Curve()
    # plotConvResults()
    Line_PlotTesults()  # alg
    Plots_Results()  # net
    Table()
    # Plot_Confusion()
