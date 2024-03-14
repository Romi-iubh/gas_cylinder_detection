#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:23:33 2024

@author: deeplearning
"""
import cv2
import numpy as np
import tensorflow as tf
import math
import colorsys
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    tqdm_installed = True
except ImportError:
    tqdm_installed = False


def visualize_prediction(predictions, original_img, resized_img, resize_info):
    boxes = predictions['detection_boxes'][0] #bounding boxes
    masks = predictions['detection_masks'][0] #instance segmentation masks
    scores = predictions['detection_scores'][0] #confidence score for each preduction
    classes = predictions['detection_classes'][0] #prediction classes (there is only one class)
    nb_detection = predictions['num_detections'][0].numpy() #number of object detected
    
    #reframe the mask to align them with the input image
    mask_proba_treshold = 0.3 # treshold for a pixel probability to be considered true
    masks_reframed = reframe_box_masks_to_image_masks(
              masks, boxes/256.0,
              np.shape(resized_img.numpy())[1],
              np.shape(resized_img.numpy())[2])
    #convert from proba to binary masks
    masks_reframed = tf.cast(
        masks_reframed > mask_proba_treshold,
        np.uint8)
    predictions['masks_reframed'] = masks_reframed.numpy()
    
    
    #=============================================================================
    #filter predictions
    #=============================================================================
    #Parameters for predidction filtering
    iou_threshold = 0.7
    score_treshold = 0.8
    
    
    #Non maximum suppresison algorithm executed based on the parameters
    idx_from_NMS = tf.image.non_max_suppression(
        boxes, scores, max_output_size=nb_detection, iou_threshold=iou_threshold)
    idx_from_NMS  = idx_from_NMS[None, :]
    idx_from_score = tf.where(scores > score_treshold)
    idx_from_score = tf.reshape(idx_from_score, 
                                (1,tf.shape(idx_from_score)[0]))
    idx_from_score = tf.cast(idx_from_score, tf.int32)
    
    #determine the indices of the predidctions to keep
    filtered_indices = tf.sets.intersection(idx_from_NMS , idx_from_score)
    filtered_indices = tf.sparse.to_dense(filtered_indices)[0]
    
    #Filter the prediction
    filtered_boxes = tf.gather(boxes, filtered_indices)
    filtered_scores= tf.gather(scores, filtered_indices)
    filtered_classes = tf.gather(classes, filtered_indices)
    filtered_masks = predictions['masks_reframed'][filtered_indices]
    
    #Upscale the mask and bounding boxe from 256x256 to the original image size
    masks_to_plot = upscale_img(filtered_masks, resize_info)
    bboxes_to_plot = upscale_bbox(filtered_boxes, resize_info)
    
    
    #=============================================================================
    #Visualizing predictions
    #=============================================================================
    
    batch_size, _, _ = masks_to_plot.shape
    height, width, _ = original_img.shape

    overlay_colors = generate_colors(batch_size)
    
    alpha = 0.3  # Transparency factor
    
    # Convert image to float32 for processing
    image_float = np.copy(original_img).astype(np.float32)
    overlay_combined = np.zeros_like(image_float, dtype=np.float32)
    
    if tqdm_installed:
        for i in tqdm(range(batch_size), desc="Applying masks", unit="image"):
            mask = masks_to_plot[i].astype(bool)
            overlay_color = np.array(overlay_colors[i % len(overlay_colors)], dtype=np.float32)
            
            # Create an overlay image for this mask
            overlay = np.zeros_like(image_float, dtype=np.float32)
            overlay[..., 0] = mask * overlay_color[0]  # Red channel
            overlay[..., 1] = mask * overlay_color[1]  # Green channel
            overlay[..., 2] = mask * overlay_color[2]  # Blue channel
            overlay_combined = np.where(overlay_combined == 0, overlay, overlay_combined)
        # Blend the overlay with the original image
        image_float = (1 - alpha) * image_float + alpha * overlay_combined
            
    else:
        for i in range(batch_size): 
            mask = masks_to_plot[i].astype(bool)
            overlay_color = np.array(overlay_colors[i % len(overlay_colors)], dtype=np.float32)
            
            # Create an overlay image for this mask
            overlay = np.zeros_like(image_float, dtype=np.float32)
            overlay[..., 0] = mask * overlay_color[0]  # Red channel
            overlay[..., 1] = mask * overlay_color[1]  # Green channel
            overlay[..., 2] = mask * overlay_color[2]  # Blue channel
            overlay_combined = np.where(overlay_combined == 0, overlay, overlay_combined)
        # Blend the overlay with the original image
        image_float = (1 - alpha) * image_float + alpha * overlay_combined
    
    # Ensure the blended image is in the valid range
    blended_image = np.clip(image_float, 0, 255).astype(np.uint8)
    
    
    #Contours and text !
    #==============================================
    blended_image2 = np.copy(blended_image)
    if tqdm_installed:
        for i in tqdm(range(batch_size), desc="Counting gas cylinders", unit="image"):
            mask = masks_to_plot[i].astype(bool)
            bbox = bboxes_to_plot[i]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_concat = np.concatenate([c for c in contours], axis=0)
            
            ymin, xmin, ymax, xmax= bbox.numpy()
            # Calculate the position for placing the text (center of the bounding box)
            text_position = (int(xmin + (xmax - xmin)// 2) , int(ymin + 0.15* (ymax-ymin)) )
            # Add text to the image
            r = 1 if (width * 0.033)<1 else int((width * 0.035)) # radius
            cv2.circle(blended_image2, tuple(text_position), r, (255, 255, 255), -1, cv2.LINE_AA)
            # Get the size of the text
            text = str(i + 1) 
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1 if (width * 0.0015)<1 else (width * 0.0015)
            thickness = 1 if (width * 0.004)<1 else int(width * 0.004)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Calculate the starting position to center the text
            start_x = int(text_position[0] - text_size[0] / 2)
            start_y = int(text_position[1] + text_size[1] / 2)
            # Draw the text inside the circle
            cv2.putText(blended_image2, text, (start_x, start_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    else:
        for i in range(batch_size): 
            mask = masks_to_plot[i].astype(bool)
            bbox = bboxes_to_plot[i]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_concat = np.concatenate([c for c in contours], axis=0)
            
            ymin, xmin, ymax, xmax= bbox.numpy()
            # Calculate the position for placing the text (center of the bounding box)
            text_position = (int(xmin + (xmax - xmin)// 2) , int(ymin + 0.15* (ymax-ymin)) )
            # Add text to the image
            r = 1 if (width * 0.033)<1 else int((width * 0.025)) # radius
            cv2.circle(blended_image2, tuple(text_position), r, (255, 255, 255), -1, cv2.LINE_AA)
            # Get the size of the text
            text = str(i + 1) 
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8 if (width * 0.001)<1 else (width * 0.001)
            thickness = 1 if (width * 0.0048)<1 else int(width * 0.0048)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Calculate the starting position to center the text
            start_x = int(text_position[0] - text_size[0] / 2)
            start_y = int(text_position[1] + text_size[1] / 2)
            # Draw the text inside the circle
            cv2.putText(blended_image2, text, (start_x, start_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    
    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(blended_image2)
    plt.axis('off')
    plt.show()
    
def generate_colors(num_colors):
    # Generate distinct HSV colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    # Convert HSV to RGB
    rgb_colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors

def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, resize_info = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image, resize_info

def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=1,
                          method=tf.image.ResizeMethod.BILINEAR):
    """Resizes the input image to output size (RetinaNet style).
      
    Resize and pad images given the desired output size of the image and
    stride size.
      
    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
       the largest rectangle to be bounded by the rectangle specified by the
       `desired_size`.
    2. Pad the rescaled image to the padded_size.
      
    Args:
      image: a `Tensor` of shape [height, width, 3] representing an image.
      desired_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the desired actual output image size.
      padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size. Padding will be applied
        after scaling the image to the desired_size. Can be None to disable
        padding.
      aug_scale_min: a `float` with range between [0, 1.0] representing minimum
        random scale applied to desired_size for training scale jittering.
      aug_scale_max: a `float` with range between [1.0, inf] representing maximum
        random scale applied to desired_size for training scale jittering.
      seed: seed for random scale jittering.
      method: function to resize input image to scaled image.
      
    Returns:
      output_image: `Tensor` of shape [height, width, 3] where [height, width]
        equals to `output_size`.
      image_info: a 2D `Tensor` that encodes the information of the image and the
        applied preprocessing. It is in the format of
        [[original_height, original_width], [desired_height, desired_width],
         [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
        desired_width] is the actual scaled image size, and [y_scale, x_scale] is
        the scaling factor, which is the ratio of
        scaled dimension / original dimension.
    """
    with tf.name_scope('resize_and_crop_image'):
      image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
      
      random_jittering = (
          isinstance(aug_scale_min, tf.Tensor)
          or isinstance(aug_scale_max, tf.Tensor)
          or not math.isclose(aug_scale_min, 1.0)
          or not math.isclose(aug_scale_max, 1.0)
      )
      
      if random_jittering:
        random_scale = tf.random.uniform(
            [], aug_scale_min, aug_scale_max, seed=seed)
        scaled_size = tf.round(random_scale * tf.cast(desired_size, tf.float32))
      else:
        scaled_size = tf.cast(desired_size, tf.float32)
      
      scale = tf.minimum(
          scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
      scaled_size = tf.round(image_size * scale)
      
      # Computes 2D image_scale.
      image_scale = scaled_size / image_size
      
      # Selects non-zero random offset (x, y) if scaled image is larger than
      # desired_size.
      if random_jittering:
        max_offset = scaled_size - tf.cast(desired_size, tf.float32)
        max_offset = tf.where(
            tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
        offset = max_offset * tf.random.uniform([2,], 0, 1, seed=seed)
        offset = tf.cast(offset, tf.int32)
      else:
        offset = tf.zeros((2,), tf.int32)
      
      scaled_image = tf.image.resize(
          image, tf.cast(scaled_size, tf.int32), method=method)
      
      if random_jittering:
        scaled_image = scaled_image[
            offset[0]:offset[0] + desired_size[0],
            offset[1]:offset[1] + desired_size[1], :]
      
      output_image = scaled_image
      if padded_size is not None:
        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1])
      
      image_info = tf.stack([
          image_size,
          tf.cast(desired_size, dtype=tf.float32),
          image_scale,
          tf.cast(offset, tf.float32)])
      return output_image, image_info


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
    """Transforms the box masks back to full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
      box_masks: A tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
      resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
        'bilinear' is only respected if box_masks is a float.
    Returns:
      A tensor of size [num_masks, image_height, image_width] with the same dtype
      as `box_masks`.
    """
    resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
      """The default function when there are more than 0 box masks."""
    
      num_boxes = tf.shape(box_masks)[0]
      box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    
      resized_crops = tf.image.crop_and_resize(
          image=box_masks_expanded,
          boxes=reframe_image_corners_relative_to_boxes(boxes),
          box_indices=tf.range(num_boxes),
          crop_size=[image_height, image_width],
          method=resize_method,
          extrapolation_value=0)
      return tf.cast(resized_crops, box_masks.dtype)
    
    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
    return tf.squeeze(image_masks, axis=3)


def upscale_img(images, resize_info):
    '''
    images: a batch of image. Shape is [batch_size, with, height, channels]
    
    resize_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
    '''
    #images = copy.deepcopy(result['detection_masks_reframed'])
    if (images.shape[-1] !=1) & (images.shape[-1] !=3): #add the channel if not existing
        images = tf.expand_dims(images, axis=-1)
    original_height, original_width = resize_info[0].numpy()
    desired_height, desired_width = resize_info[1].numpy()
    y_scale, x_scale = resize_info[2].numpy()
    y_offset, x_offset = resize_info[3].numpy()
    
    intermediate_height = min(round(original_height*y_scale), 256)
    intermediate_width = min(round(original_width*x_scale), 256)
    
    
    croped_imgs = images[:, :intermediate_height, :intermediate_width, :]
    
    #resized_upscaled_imgs = []
    croped_imgs = [img for img in croped_imgs]
    
    
    upscaled_imgs_list = []
    for img in croped_imgs :
        rescaled_im = cv2.resize(img.numpy(), 
                                 (int(original_width), int(original_height )),
                                 cv2.INTER_CUBIC)
        upscaled_imgs_list.append(rescaled_im)
    
    
    upscaled_imgs = np.stack(upscaled_imgs_list)
    
    return upscaled_imgs

def upscale_bbox(bboxes, resize_info):

    ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=-1)
    original_height, original_width = resize_info[0].numpy()
    desired_height, desired_width = resize_info[1].numpy()
    y_scale, x_scale = resize_info[2].numpy()
    y_offset, x_offset = resize_info[3].numpy()
    
    #find the size that the image after resizing and before being padded
    intermediate_height = min(round(original_height*y_scale), 256)
    intermediate_width = min(round(original_width*x_scale), 256)
    
    #clip ymax and xmax to the size that the image had before being padded
    xmax = tf.where(xmax > intermediate_width, intermediate_width, xmax)
    ymax = tf.where(ymax > intermediate_height, intermediate_height, ymax)
    
    #Calculate the coordinate in the upscale images
    ymin = ymin/y_scale
    xmin = xmin/x_scale
    ymax = ymax/y_scale
    xmax = xmax/x_scale
    
    upscaled_bboxes = tf.stack([ymin, xmin, ymax, xmax ], axis=-1)
    upscaled_bboxes = tf.round(upscaled_bboxes)
    #print(upscaled_bboxes)
    return upscaled_bboxes


def reframe_image_corners_relative_to_boxes(boxes):

    """Reframe the image corners ([0, 0, 1, 1]) to be relative to boxes.
    The local coordinate frame of each box is assumed to be relative to
    its own for corners.
    Args:
      boxes: A float tensor of [num_boxes, 4] of (ymin, xmin, ymax, xmax)
        coordinates in relative coordinate space of each bounding box.
    Returns:
      reframed_boxes: Reframes boxes with same shape as input.
    """
    ymin, xmin, ymax, xmax = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    
    height = tf.maximum(ymax - ymin, 1e-4)
    width = tf.maximum(xmax - xmin, 1e-4)
    
    ymin_out = (0 - ymin) / height
    xmin_out = (0 - xmin) / width
    ymax_out = (1 - ymin) / height
    xmax_out = (1 - xmin) / width
    return tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)
