import os
import cv2
import numpy as np
import pandas as pd

def read_images(img_directory):
    """
    Reads all the images in the given directory and returns a dataframe 
    with the image data and the image name.
    
    Parameters
    ----------
    img_directory : str
        The path to the directory containing the images.
    
    Returns
    -------
    data : pd.DataFrame
        A dataframe with the images and their corresponding names.
    """
    images = []
    img_name = []
    for filename in os.listdir(img_directory):
        img_path = os.path.join(img_directory, filename)
        if not os.path.isfile(img_path):
            raise ValueError(f"The path {img_path} is not a file.")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read the image at {img_path}.")
        img_name.append(filename)
        
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize the image to the desired size
        img = cv2.resize(img, (224, 224))
        
        # Append the image to the list of images
        images.append(img)
    
    # Create a dataframe with the image data and the image name
    data = pd.DataFrame({'image': images, 'name': img_name})
    
    # Return the dataframe
    return data

def add_labels(data, labels_csv_path):
    """
    Adds a label column to the dataframe by reading the labels from a CSV file.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the images and their names.
    labels_csv_path : str
        The path to the CSV file containing the labels.
    
    Returns
    -------
    data : pd.DataFrame
        The dataframe with an added label column.
    """
    labels_df = pd.read_csv(labels_csv_path)
    data = data.merge(labels_df, left_on='name', right_on='filename')
    data = data.drop(columns=['filename'])
    return data

# Example usage
# img_directory = 'path/to/images'
# labels_csv_path = 'data/labels/labels.csv'
# data = read_images(img_directory)
# data = add_labels(data, labels_csv_path)
