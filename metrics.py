import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
smooth=1e-15
def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice Coefficient for image segmentation tasks.
    
    Args:
        y_true (tensor): Ground truth mask.
        y_pred (tensor): Predicted mask.
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        tensor: Dice coefficient score.
    """
    # Flatten the tensors
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    
    # Compute the intersection
    intersection = tf.reduce_sum(y_true * y_pred)
    
    # Compute the Dice coefficient
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    
    return dice

def dice_loss(y_true, y_pred):
    """
    Compute the Dice Loss, which is 1 - Dice Coefficient.
    
    Args:
        y_true (tensor): Ground truth mask.
        y_pred (tensor): Predicted mask.
    
    Returns:
        tensor: Dice loss value.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)