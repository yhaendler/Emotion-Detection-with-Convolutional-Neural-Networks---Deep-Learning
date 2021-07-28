#####################################################################################

# function to do data augmentation
def data_augmentation(happy_input_folder='../datasets/happy', happy_output_folder='../datasets/augmented/happy',
                      neutral_input_folder='../datasets/neutral', neutral_output_folder='../datasets/augmented/neutral'):
    
    ## For happy faces ##

    # specify working directory
    chosen_images_happy = os.listdir(happy_input_folder)

    # data augmentation
    gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)

    # loop to take each image, augment it and save the new images in a folder
    for file_happy in chosen_images_happy:
        # image path
        image_path = happy_input_folder + '/' + file_happy
        #expand the dimensions so that the image is compatible for how we'll use it later
        image = np.expand_dims(plt.imread(image_path),0)
        # augment the data creating additional images
        aug_iter = gen.flow(image, save_to_dir=happy_output_folder, save_prefix='aug-image-', save_format = 'jpeg')
        aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    
    ## For neutral faces ##
    chosen_images_neutral = os.listdir(neutral_input_folder)
    
    gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)
    
    for file_neutral in chosen_images_neutral:
        # image path
        image_path = neutral_input_folder + '/' + file_neutral
        #expand the dimensions so that the image is compatible for how we'll use it later
        image = np.expand_dims(plt.imread(image_path),0)
        # augment the data creating additional images
        aug_iter = gen.flow(image, save_to_dir=neutral_output_folder, save_prefix='aug-image-', save_format = 'jpeg')
        aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]

#####################################################################################

# function to produce a table with different classification metrics
def model_scores(y_true, y_pred):
    
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    
    # confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # dictionary
    scores_dict = {}
    scores_dict['accuracy'] = (tn+tp)/(tn+fp+fn+tp)
    scores_dict['misclassification'] = 1 - ((tn+tp)/(tn+fp+fn+tp))
    scores_dict['specificity'] = tn/(tn+fp)
    scores_dict['recall_sensitivity'] = tp/(tp+fn)
    scores_dict['precision'] = tp/(tp+fp)
    scores_dict['f1_score'] = 2 * (scores_dict['precision']*scores_dict['recall_sensitivity'])/(scores_dict['precision']+scores_dict['recall_sensitivity'])
    
    scores_df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=['test_data_model_metrics'])
    
    return scores_df

#####################################################################################

# function to produce a table with images about which the NN predicts if they are 'happy' or 'neutral'
import numpy as np
import pandas as pd

def test_predictions(test_data, predictions):
    
    summary_dict = {}
    summary_dict['file'] = []
    summary_dict['predicted_label'] = []
    summary_dict['race'] = []
    summary_dict['age'] = []

    for pred, filename in zip(np.round(predictions),  test_data.filenames):

        # adding file name to dictionary
        summary_dict['file'].append(filename)

        # adding predicted label to dictionary
        if pred == 0:
            summary_dict['predicted_label'].append('happy')
        else:
            summary_dict['predicted_label'].append('neutral')

        # adding race info to dictionary
        if 'asian' in filename:
            summary_dict['race'].append('asian')
        elif 'dark' in filename:
            summary_dict['race'].append('dark')
        else:
            summary_dict['race'].append('white')

        # adding age info to dictionary
        if 'babies' in filename:
            summary_dict['age'].append('babies')
        elif 'children' in filename:
            summary_dict['age'].append('children')
        elif 'adults' in filename:
            summary_dict['age'].append('adults')
        else:
            summary_dict['age'].append('elderly')

    summary_df = pd.DataFrame(summary_dict, columns=summary_dict.keys())

    # creating a result column
    summary_df['result'] = np.nan

    # if the model is correct, result = 1, otherwise result = 0
    for row in range(summary_df.shape[0]):
        if '_happy_' in summary_df.loc[row, 'file'] and summary_df.loc[row, 'predicted_label'] == 'happy':
            summary_df.loc[row, 'result'] = int(1)
        elif '_neutral_' in summary_df.loc[row, 'file'] and summary_df.loc[row, 'predicted_label'] == 'neutral':
            summary_df.loc[row, 'result'] = int(1)
        else: 
            summary_df.loc[row, 'result'] = int(0)
    
    summary_df['result'] = summary_df['result'].astype(int)    
    
    return summary_df

#####################################################################################

# function to plot confusion matrix from a confusion matrix object, rather than from X and y (like in sklearn's built-in function) - taken from https://deeplizard.com/learn/video/bfQBPNDy5EM

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#####################################################################################

# function to plot images from an image array - taken from Tensorflow's webpage
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
#####################################################################################
