import matplotlib.pyplot as plt
import numpy as np
import itertools
import pygal
from pygal.style import Style
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def get_precision_recall_f1_score(classificationReport):
    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')
    
    t = lines[-3].strip().split()
    #print(t[0]+' '+t[1])
    precision = float(t[2])
    recall = float(t[3])
    f1_score = float(t[4])

    return precision, recall, f1_score

def plot_classification_report(classificationReport,
                               output_file, 
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names, precision, recall, f1_score = [], [], [], [], [], [], []
    for line in lines[1:-5]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        #v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        #plotMat.append(v)
        precision.append(float(t[1]))
        recall.append(float(t[2]))
        f1_score.append(float(t[3]))
        
 

    # create plot

    fig, ax = plt.subplots()
    index = np.arange(len(classes)*2,step=2)  
    bar_width = 0.5
    opacity = 0.8

    
    rects1 = plt.bar(index, precision, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#4169E1',
    hatch="////",
    label='Precision')

    rects2 = plt.bar(index + bar_width, recall, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#228B22',
    hatch="XXX",
    label='Recall')

    rects3 = plt.bar(index + 2*bar_width, f1_score, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#FF8C00',
    hatch="***",
    label='F1-score')


    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
    

    plt.xticks(index + bar_width, class_names , rotation=0)
    #plt.legend(loc="upper right")
    #plt.tight_layout()
    plt.ylabel('Precision / Recall / F1-score')
    plt.grid(axis='y',linestyle='dashed', linewidth=0.5) #linestyle='-', linewidth=2 
    #plt.savefig('../results/'+name)


    plt.xlabel('Fault classes')
    
    #plt.title(title)
    
    #fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_file, dpi=100)
    plt.show()

    plt.close()

    return output_file


def plot_cepe_category_classification_report(classificationReport,
                                             output_file, 
                                             title='CE-PE routing faults classification report',
                                             cmap='RdBu'):
    
    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')
    del lines[0] 

    

    categories = ['VRF', 'Interfaces', 'eBGP routing', 'Static routing']  # 'Valid conf.',  
    classes = {
        # 'Valid conf.': [0],
        'VRF': [1, 2],
        'Interfaces': [5, 6, 14, 15, 23, 24, 25, 29],
        'eBGP routing': [3, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 26],
        'Static routing': [18, 19, 20, 21, 22, 27, 28]
    }

    precisions, recalls, f1_scores = [], [], []

    for cat in categories:
        precision, recall, f1_score = 0, 0, 0
        for f in classes[cat]:
            t = lines[f].strip().split() 
            precision+= float(t[1])
            recall+= float(t[2])
            f1_score+= float(t[3])
        precisions.append(precision/len(classes[cat]))
        recalls.append(recall/len(classes[cat]))  
        f1_scores.append(f1_score/len(classes[cat]))


    
    # create plot

    fig, ax = plt.subplots()
    index = np.arange(len(categories)*2,step=2)  
    bar_width = 0.5
    opacity = 0.8

    rects1 = plt.bar(index, precisions, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#4169E1',
    hatch="////",
    label='Precision')

    rects2 = plt.bar(index + bar_width, recalls, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#228B22',
    hatch="XXX",
    label='Recall')

    rects3 = plt.bar(index + 2*bar_width, f1_scores, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#FF8C00',
    hatch="***",
    label='F1-score')


    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
    

    plt.xticks(index + bar_width, categories , rotation=0, fontsize=8) # class_names
    #plt.legend(loc="upper right")
    #plt.tight_layout()
    plt.ylabel('Precision / Recall / F1-score')
    plt.grid(axis='y',linestyle='dashed', linewidth=0.5) #linestyle='-', linewidth=2 
    #plt.savefig('../results/'+name)


    plt.xlabel('Fault categories')
    #plt.title(title)
    
    #fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_file, dpi=100)
    plt.show()

    plt.close()

    return output_file



def plot_f1_score(pes_f1_scores, cepe_f1_score, classes, xlabel, output_file):
    
    
    #classes = classes
    #class_names = classes 

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(classes)*2,step=2)
    bar_width = 0.5
    opacity = 0.8


    rects1 = plt.bar(index, pes_f1_scores, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#FF8C00',
    hatch="***",
    label='PE-PE routing model')

    rects2 = plt.bar(index + bar_width, cepe_f1_score, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#FF8C00',
    hatch="xxx",
    label='CE-PE routing model')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
    
    plt.xticks(index + 0.5 * bar_width, classes , rotation=0)
    #plt.legend(loc="upper right")
    #plt.tight_layout()
    plt.ylabel('F1-score')

    plt.xlabel(xlabel)

    plt.grid(axis='y',linestyle='dashed', linewidth=0.5) #linestyle='-', linewidth=2 
       
    #fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_file, dpi=100)
    plt.show()
    plt.close()

    return output_file

def plot_general_report(precisions, recalls, f1_scores, classes, xlabel, output_file):

    #classes = classes
    #class_names = classes 

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(classes)*2,step=2)
    bar_width = 0.5
    opacity = 0.8


    rects1 = plt.bar(index, precisions, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#4169E1',
    hatch="////",
    label='Precision')

    rects2 = plt.bar(index + bar_width, recalls, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#228B22',
    hatch="XXX",
    label='Recall')

    rects3 = plt.bar(index + 2*bar_width, f1_scores, bar_width,
    alpha=opacity,
    color='white',
    edgecolor='#FF8C00',
    hatch="***",
    label='F1-score')

    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
    

    plt.xticks(index + bar_width, classes , rotation=0)
    #plt.legend(loc="upper right")
    #plt.tight_layout()
    plt.ylabel('Precision / Recall / F1-score')

    plt.xlabel(xlabel)

    plt.grid(axis='y',linestyle='dashed', linewidth=0.5) #linestyle='-', linewidth=2 
       
    #fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_file, dpi=100)
    plt.show()
    plt.close()

    return output_file