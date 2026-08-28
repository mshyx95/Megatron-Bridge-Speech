import os
import json
import random
import numpy as np
from matplotlib import path
from multiprocessing import Process
from typing import Callable
import argparse
import time
import hashlib


  
def hash_string(input_string):  
    # Create a new SHA256 hash object  
    hasher = hashlib.sha256()  
  
    # Encode the string to bytes, as the hash function requires  
    byte_string = input_string.encode()  
  
    # Hash the byte string  
    hasher.update(byte_string)  
  
    # Get the hexadecimal representation of the hash  
    hash_string = hasher.hexdigest()  
  
    return hash_string[0:32]


def process_dense_caption(data, r):
    #{"dense_captions": {"detections": [{"bbox": [0.12468118965625763, 0.7580925822257996, 0.30859342217445374, 0.9984610676765442], "score": 0.5086951851844788, "fl_score": 0.31135857105255127, "category": "bench"}, {"bbox": [0.5568585395812988, 0.13083678483963013, 0.7950448989868164, 0.2539951801300049], "score": 0.4580952823162079, "fl_score": 0.30611419677734375, "category": "mechanical fan"}]}}
    caption_list = data["dense_captions"]["detections"]
    if len(caption_list) < 1:
        return  ""
    else:
        string_list = []
        string = f"Instruct: List all the objects in this image <|img_{r}|>.\n"
        lst = [cc['category'] for cc in caption_list]
        string += f"Output: The objects are {', '.join(lst)}."
        string_list.append(string)
        for cc in caption_list:
            string = f"Instruct: What is the bounding box of the {cc['category']} in this image <|img_{r}|>?\n"

            bbx = cc['bbox']
            bbx_axis = []
            for bb in bbx:
                bbx_axis.append(round(bb, 2))

            string += f"Output: The bounding box is {bbx_axis}.\n"
            string_list.append(string)
            
            
            string = f"Instruct: What object is in the bounding box of form {bbx_axis} in this image <|img_{r}|>?\n"
            string += f"Output: The object is {cc['category']}\n"
            string_list.append(string)
        '''
        string = f"Instruct: List all the objects and their bounding boxes in this image <|img_{r}|>.\nOutput: Sure, here are all the objects and their bounding boxes:\n"
        for i, cc in enumerate(caption_list):
            bbx = cc['bbox']
            bbx_axis = []
            for bb in bbx:
                bbx_axis.append(round(bb, 2))
            
            string += f"{i}. Object name: {cc['category']}, bounding box: {bbx_axis}\n"

            
        string_list.append(string)
        '''
        return "\n<|endoftext|>\n".join(string_list)


def process_od_caption(data, r):
    caption_list = data["detections"]
    if len(caption_list) < 1:
        return  ""
    else:
        string_list = []
        #string_list = []
        string = f"Instruct: List all the objects in this image <|img_{r}|>.\n"
        lst = [cc['category'] for cc in caption_list]
        string += f"Output: The objects are {', '.join(lst)}."
        string_list.append(string)
        for cc in caption_list:
            string = f"Instruct: What is the bounding box of the {cc['category']} in this image <|img_{r}|>?\n"

            bbx = cc['bbox']
            bbx_axis = []
            for bb in bbx:
                bbx_axis.append(round(bb, 2))

            string += f"Output: The bounding box is {bbx_axis}.\n"
            string_list.append(string)
            
            
            string = f"Instruct: What object is in the bounding box of form {bbx_axis} in this image <|img_{r}|>?\n"
            string += f"Output: The object is {cc['category']}\n"
            string_list.append(string)
        '''
        string = f"Instruct: List all the objects and their bounding boxes in this image <|img_{r}|>.\nOutput: Sure, here are all the objects and their bounding boxes:\n"
        for i, cc in enumerate(caption_list):
            bbx = cc['bbox']
            bbx_axis = []
            for bb in bbx:
                bbx_axis.append(round(bb, 2))
            
            string += f"{i}. Object name: {cc['category']}, bounding box: {bbx_axis}\n"

            
        string_list.append(string)
        '''
        return "\n<|endoftext|>\n".join(string_list)




def box_to_string(boxes):
    bbx_axis = []
    for bb in boxes:
        bbx_axis.append(round(bb, 2))

    return str(bbx_axis)

def process_regular_caption(data, r):
    #{"localized_narratives": {"captions": ["The image shows an empty room with a wooden bench on the left side, a door and a window in the middle, and lights and a fan on the ceiling. It is a 700 sq ft 2 bhk independent floor for rent in Kharghar, with a layout of 16000 sq ft."], "parsed_anns": [{"num_parsed_tokens": 37, "object_complexity": 5, "action_complexity": 2, "actions": ["shows"], "num_actions": 1, "attributes": ["empty", "wooden", "left", "700", "sq", "2", "bhk", "independent", "16000", "sq"], "num_attributes": 10, "objects": ["image", "room", "bench", "side", "door", "window", "middle", "lights", "fan", "ceiling", "It", "ft", "floor", "rent", "Kharghar", "layout", "ft"], "num_objects": 17, "proper_nouns": ["Kharghar"], "num_proper_nouns": 1, "first_letter_capitalized": ["It", "Kharghar"], "num_first_letter_capitalized": 2}], "type": "syn"}}
    caption = data['localized_narratives']['captions']
    string_list = []
    for cc in caption:
        string = f"Here is an image <|img_{r}|>.\n"
        string += f"Instruction: What is in this image?\nOutput: {cc}\n"
    string_list.append(string)
    #string = f"Instruct: What is in this image <|img_{r}|>?\n"
    #string += f"Output: {caption}\n"
    return "\n<|endoftext|>\n".join(string_list)


def process_grouding_caption(data, r, max_rounds=None):
    #{"grounding": {"caption": "The image shows an empty room with a wooden bench on the left side, a door and a window in the middle, and lights and a fan on the ceiling. It is a 700 sq ft 2 bhk independent floor for rent in Kharghar, with a layout of 16000 sq ft.", "restored_caption": "the image shows an empty room with a wooden bench on the left side, a door and a window in the middle, and lights and a fan on the ceiling. it is a 700 sq ft 2 bhk independent floor for rent in kharghar, with a layout of 16000 sq ft.", "phrases": ["an empty room", "a wooden bench", "a door", "a window", "lights", "a fan", "the ceiling", "2 bhk independent floor"], "phrases_span": [[16, 29], [35, 49], [68, 74], [79, 87], [107, 113], [118, 123], [127, 138], [158, 181]], "boxes": [[0.00018781423568725586, 0.0004341006278991699, 1.0001716613769531, 1.0001022815704346], [0.10077780485153198, 0.0010442882776260376, 0.9994497895240784, 0.30059462785720825], [0.12226948142051697, 0.7842037677764893, 0.9993935823440552, 1.0001920461654663], [0.6980388164520264, 0.4024566411972046, 0.8716299533843994, 0.8408969640731812], [0.8464334011077881, 0.36851781606674194, 0.9709113836288452, 0.9100267291069031], [0.43097198009490967, 0.4074723720550537, 0.6496894359588623, 0.6491833925247192], [0.12106059491634369, 0.7512788772583008, 0.3132143020629883, 0.999945878982544], [0.545217752456665, 0.11298266053199768, 0.8010259866714478, 0.271740198135376], [0.20688103139400482, 0.07511521875858307, 0.25249066948890686, 0.12250833213329315]], "phrases_to_boxes": [[0], [6], [3, 4], [5], [8], [7], [1], [2]], "boxes_to_phrases": [[0], [6], [7], [2], [2], [3], [1], [5], [4]], "phrase_box_index_to_score": [[[5, 7], 0.82], [[3, 5], 0.71], [[2, 3], 0.58], [[1, 6], 0.82], [[0, 0], 0.34], [[6, 1], 0.6], [[2, 4], 0.28], [[7, 2], 0.41], [[4, 8], 0.31]], "bbox_format": "xyxy", "bbox_normalized": true, "cased_caption": "The image shows an empty room with a wooden bench on the left side, a door and a window in the middle, and lights and a fan on the ceiling. It is a 700 sq ft 2 bhk independent floor for rent in Kharghar, with a layout of 16000 sq ft.", "cased_phrases_span": [[16, 29], [35, 49], [68, 74], [79, 87], [107, 113], [118, 123], [127, 138], [158, 181]], "cased_phrases": ["an empty room", "a wooden bench", "a door", "a window", "lights", "a fan", "the ceiling", "2 bhk independent floor"]}}
    phrases = data['grounding']['phrases']
    boxes =  data['grounding']['boxes']
    phrases_to_boxes = data['grounding']['phrases_to_boxes']
    boxes_to_phrases = data['grounding']['boxes_to_phrases']
    string_list = []

    '''
    string = f"Here is an image <|img_{r}|>.\n"
    string += "Instruct: List the objects and their bounding boxes in this image.\nOutput: Sure, here are some of the objects and their bounding boxes:\n"
    for i, pp in enumerate(phrases):
        bxs = phrases_to_boxes[i]
        bx = []
        for bb in bxs:
            bx.append(box_to_string(boxes[bb]))
        bx = " and ".join(bx)

        string += f"{i}. Object name: {phrases[i]}, bounding box: {bx}.\n"
    string_list.append(string)
    '''
    #string = f"Instruct: What is in this image?\nOutput: {data['grounding']['caption']}"
    #string_list.append(string)
    #for i, bb in enumerate(boxes):

    string = f"Instruct: List all the objects in this image <|img_{r}|>.\n"
    lst = [cc for cc in phrases]
    string += f"Output: The objects are {', '.join(lst)}."
    string_list.append(string)
    
    for i, pp in enumerate(phrases):
        string = f"Here is an image <|img_{r}|>.\n"
        
        bxs = phrases_to_boxes[i]
        bx = []
        for bb in bxs:
            bx.append(box_to_string(boxes[bb]))
        bx = " and ".join(bx)

        string += f"Instruct: Tell me what is the bounding box of the object **{pp}** in this image.\nOutput: Sure, the bounding box is {bx}\n"

        string_list.append(string)
    
    if max_rounds is not None:
        # random choice of max_rounds
        random.shuffle(string_list)
        string_list = string_list[:max_rounds]

    return  "\n<|endoftext|>\n".join(string_list)


def process_text_caption(data, r):
    #{"original_height": 300, "original_width": 400, "synthetic_captions": ["residential house for rent in thanthura"], "captions": ["gallery cover image of 700 sq ft 2 bhk independent floor for rent in hsr layout for 16000"]}
    caption = data['synthetic_captions']
    string = f"Here is an image <|img_{r}|>.\n"
    string += f"Instruction: Generate a caption for this image.\nOutput: Some possible captions are: {caption}.\n"
    return string


def process_gptv_caption(data, r):
    #{"gpt_v_image_paragraph_caption": {"captions": ["The image shows an empty room with white walls and marble flooring. The room has a ceiling with recessed lighting and a ceiling fan. There is a window on the wall with white shutters, and a wooden bench on the left side of the room. On the right side, there is a black door leading to another room. The floor is made of marble tiles, and there are a few items scattered around the room, including a small plant in a pot on the floor."]}}
    caption = data['gpt_v_image_paragraph_caption']['captions']
    string = f"Here is an image <|img_{r}|>.\n"
    string += f"Instruction: Give me a detailed desciption of this image.\nOutput: {caption}\n"
    return string

def process_ocr_caption(data, r):
    '''
        {"read": {"apiVersion": "internal-latest", "modelId": "prebuilt-read-2022-04-30", "stringIndexType": "textElements", "content": "HAVEN HOUSE\nRENEWING LIVES\nRestoring Hope\nSSOUDJRMY\nSUMMER\nG0 TEARS\nRecovery Da\nHAVED HOUSE\nJOURNAL\nDRIVE\nDonated new journals are\ngiven to individuals in therapy\nto assist in the therapeutic\nprocess\nCONTACT\nLori Richardson\nThe healing\n610-433-6181 x 214\nBe brave\nBe bold\nor Irichardson@\nBe beautol\npower of\nhaven-house.com\nwriting\nFollow us on facebook for Haven House\n& mental health awareness info\nA411 Union Boulevard, Allentown PA 18109 ~ 610-433-6181x 214", "pages": [{"pageNumber": 1, "angle": 0.1996, "width": 695, "height": 900, "unit": "pixel", "words": [{"content": "HAVEN", "boundingBox": [226, 20, 328, 20, 328, 49, 225, 49], "confidence": 0.998, "span": {"offset": 0, "length": 5}}, {"content": "HOUSE", "boundingBox": [344, 20, 454, 19, 454, 49, 344, 49], "confidence": 0.998, "span": {"offset": 6, "length": 5}}, {"content": "RENEWING", "boundingBox": [654, 27, 653, 75, 640, 75, 642, 27], "confidence": 0.899, "span": {"offset": 12, "length": 8}}, {"content": "LIVES", "boundingBox": [653, 78, 653, 102, 642, 102, 640, 78], "confidence": 0.956, "span": {"offset": 21, "length": 5}}, {"content": "Restoring", "boundingBox": [561, 28, 609, 27, 608, 42, 560, 41], "confidence": 0.877, "span": {"offset": 27, "length": 9}}, {"content": "Hope", "boundingBox": [611, 27, 641, 29, 640, 43, 610, 42], "confidence": 0.473, "span": {"offset": 37, "length": 4}}, {"content": "SSOUDJRMY", "boundingBox": [577, 40, 575, 102, 562, 101, 560, 41], "confidence": 0.077, "span": {"offset": 42, "length": 9}}, {"content": "SUMMER", "boundingBox": [27, 76, 318, 79, 318, 139, 31, 141], "confidence": 0.998, "span": {"offset": 52, "length": 6}}, {"content": "G0", "boundingBox": [597, 44, 603, 44, 601, 89, 595, 89], "confidence": 0.053, "span": {"offset": 59, "length": 2}}, {"content": "TEARS", "boundingBox": [612, 44, 645, 46, 644, 91, 610, 89], "confidence": 0.132, "span": {"offset": 62, "length": 5}}, {"content": "Recovery", "boundingBox": [592, 88, 638, 91, 638, 103, 590, 101], "confidence": 0.675, "span": {"offset": 68, "length": 8}}, {"content": "Da", "boundingBox": [641, 91, 654, 93, 654, 105, 640, 104], "confidence": 0.044, "span": {"offset": 77, "length": 2}}, {"content": "HAVED", "boundingBox": [560, 105, 604, 105, 604, 121, 560, 121], "confidence": 0.485, "span": {"offset": 80, "length": 5}}, {"content": "HOUSE", "boundingBox": [608, 105, 654, 106, 653, 122, 607, 121], "confidence": 0.865, "span": {"offset": 86, "length": 5}}, {"content": "JOURNAL", "boundingBox": [163, 142, 614, 145, 609, 227, 163, 226], "confidence": 0.998, "span": {"offset": 92, "length": 7}}, {"content": "DRIVE", "boundingBox": [208, 238, 626, 235, 629, 361, 206, 361], "confidence": 0.998, "span": {"offset": 100, "length": 5}}, {"content": "Donated", "boundingBox": [367, 370, 450, 370, 450, 400, 366, 400], "confidence": 0.998, "span": {"offset": 106, "length": 7}}, {"content": "new", "boundingBox": [456, 370, 499, 371, 498, 400, 456, 400], "confidence": 0.996, "span": {"offset": 114, "length": 3}}, {"content": "journals", "boundingBox": [509, 371, 597, 371, 597, 401, 508, 400], "confidence": 0.998, "span": {"offset": 118, "length": 8}}, {"content": "are", "boundingBox": [602, 371, 641, 371, 641, 401, 602, 401], "confidence": 0.997, "span": {"offset": 127, "length": 3}}, {"content": "given", "boundingBox": [368, 411, 420, 410, 417, 443, 366, 443], "confidence": 0.998, "span": {"offset": 131, "length": 5}}, {"content": "to", "boundingBox": [426, 410, 448, 410, 446, 443, 424, 443], "confidence": 0.998, "span": {"offset": 137, "length": 2}}, {"content": "individuals", "boundingBox": [455, 410, 566, 410, 563, 443, 452, 443], "confidence": 0.993, "span": {"offset": 140, "length": 11}}, {"content": "in", "boundingBox": [573, 410, 592, 410, 589, 444, 570, 443], "confidence": 0.998, "span": {"offset": 152, "length": 2}}, {"content": "therapy", "boundingBox": [599, 410, 685, 411, 681, 445, 596, 444], "confidence": 0.998, "span": {"offset": 155, "length": 7}}, {"content": "to", "boundingBox": [368, 450, 386, 450, 386, 479, 368, 479], "confidence": 0.996, "span": {"offset": 163, "length": 2}}, {"content": "assist", "boundingBox": [392, 450, 454, 449, 454, 480, 392, 479], "confidence": 0.998, "span": {"offset": 166, "length": 6}}, {"content": "in", "boundingBox": [461, 449, 481, 449, 481, 480, 461, 480], "confidence": 0.998, "span": {"offset": 173, "length": 2}}, {"content": "the", "boundingBox": [487, 449, 521, 449, 521, 480, 487, 480], "confidence": 0.997, "span": {"offset": 176, "length": 3}}, {"content": "therapeutic", "boundingBox": [527, 449, 653, 449, 653, 482, 527, 480], "confidence": 0.993, "span": {"offset": 180, "length": 11}}, {"content": "process", "boundingBox": [366, 496, 451, 493, 450, 518, 367, 519], "confidence": 0.998, "span": {"offset": 192, "length": 7}}, {"content": "CONTACT", "boundingBox": [555, 505, 650, 505, 650, 526, 554, 526], "confidence": 0.998, "span": {"offset": 200, "length": 7}}, {"content": "Lori", "boundingBox": [503, 531, 538, 531, 538, 556, 503, 555], "confidence": 0.993, "span": {"offset": 208, "length": 4}}, {"content": "Richardson", "boundingBox": [543, 531, 651, 533, 652, 557, 543, 556], "confidence": 0.997, "span": {"offset": 213, "length": 10}}, {"content": "The", "boundingBox": [19, 577, 102, 575, 102, 626, 19, 624], "confidence": 0.997, "span": {"offset": 224, "length": 3}}, {"content": "healing", "boundingBox": [113, 575, 296, 577, 294, 630, 113, 626], "confidence": 0.998, "span": {"offset": 228, "length": 7}}, {"content": "610-433-6181", "boundingBox": [505, 568, 625, 568, 625, 592, 504, 592], "confidence": 0.969, "span": {"offset": 236, "length": 12}}, {"content": "x", "boundingBox": [630, 568, 642, 568, 643, 592, 630, 592], "confidence": 0.661, "span": {"offset": 249, "length": 1}}, {"content": "214", "boundingBox": [647, 568, 681, 567, 682, 592, 648, 592], "confidence": 0.995, "span": {"offset": 251, "length": 3}}, {"content": "Be", "boundingBox": [366, 594, 379, 594, 378, 607, 364, 607], "confidence": 0.713, "span": {"offset": 255, "length": 2}}, {"content": "brave", "boundingBox": [382, 594, 411, 595, 411, 607, 381, 607], "confidence": 0.928, "span": {"offset": 258, "length": 5}}, {"content": "Be", "boundingBox": [366, 607, 379, 607, 378, 619, 365, 619], "confidence": 0.191, "span": {"offset": 264, "length": 2}}, {"content": "bold", "boundingBox": [382, 607, 405, 607, 404, 620, 381, 619], "confidence": 0.179, "span": {"offset": 267, "length": 4}}, {"content": "or", "boundingBox": [504, 604, 528, 604, 528, 629, 504, 629], "confidence": 0.997, "span": {"offset": 272, "length": 2}}, {"content": "Irichardson@", "boundingBox": [534, 604, 661, 604, 660, 630, 533, 629], "confidence": 0.695, "span": {"offset": 275, "length": 12}}, {"content": "Be", "boundingBox": [365, 619, 378, 619, 379, 633, 366, 632], "confidence": 0.689, "span": {"offset": 288, "length": 2}}, {"content": "beautol", "boundingBox": [381, 619, 430, 619, 429, 634, 381, 633], "confidence": 0.009, "span": {"offset": 291, "length": 7}}, {"content": "power", "boundingBox": [19, 644, 179, 636, 180, 685, 20, 687], "confidence": 0.998, "span": {"offset": 299, "length": 5}}, {"content": "of", "boundingBox": [190, 635, 246, 633, 247, 682, 190, 685], "confidence": 0.998, "span": {"offset": 305, "length": 2}}, {"content": "haven-house.com", "boundingBox": [505, 640, 664, 641, 665, 665, 504, 665], "confidence": 0.989, "span": {"offset": 308, "length": 15}}, {"content": "writing", "boundingBox": [19, 690, 195, 690, 196, 746, 18, 740], "confidence": 0.998, "span": {"offset": 324, "length": 7}}, {"content": "Follow", "boundingBox": [12, 790, 79, 791, 79, 821, 13, 821], "confidence": 0.998, "span": {"offset": 332, "length": 6}}, {"content": "us", "boundingBox": [89, 791, 115, 791, 116, 821, 90, 821], "confidence": 0.998, "span": {"offset": 339, "length": 2}}, {"content": "on", "boundingBox": [121, 791, 149, 791, 150, 821, 122, 821], "confidence": 0.998, "span": {"offset": 342, "length": 2}}, {"content": "facebook", "boundingBox": [155, 791, 258, 791, 259, 821, 156, 821], "confidence": 0.998, "span": {"offset": 345, "length": 8}}, {"content": "for", "boundingBox": [264, 791, 296, 791, 297, 821, 265, 821], "confidence": 0.996, "span": {"offset": 354, "length": 3}}, {"content": "Haven", "boundingBox": [302, 791, 371, 791, 373, 821, 303, 821], "confidence": 0.998, "span": {"offset": 358, "length": 5}}, {"content": "House", "boundingBox": [377, 791, 447, 790, 449, 821, 379, 821], "confidence": 0.998, "span": {"offset": 364, "length": 5}}, {"content": "&", "boundingBox": [15, 833, 31, 834, 31, 862, 15, 862], "confidence": 0.473, "span": {"offset": 370, "length": 1}}, {"content": "mental", "boundingBox": [39, 834, 118, 834, 118, 863, 39, 862], "confidence": 0.998, "span": {"offset": 372, "length": 6}}, {"content": "health", "boundingBox": [124, 834, 197, 834, 197, 863, 124, 863], "confidence": 0.998, "span": {"offset": 379, "length": 6}}, {"content": "awareness", "boundingBox": [203, 834, 327, 834, 328, 862, 203, 863], "confidence": 0.997, "span": {"offset": 386, "length": 9}}, {"content": "info", "boundingBox": [333, 834, 379, 833, 380, 862, 333, 862], "confidence": 0.994, "span": {"offset": 396, "length": 4}}, {"content": "A411", "boundingBox": [8, 870, 49, 870, 49, 890, 8, 890], "confidence": 0.725, "span": {"offset": 401, "length": 4}}, {"content": "Union", "boundingBox": [55, 870, 111, 870, 111, 890, 55, 890], "confidence": 0.773, "span": {"offset": 406, "length": 5}}, {"content": "Boulevard,", "boundingBox": [119, 870, 233, 869, 232, 890, 119, 890], "confidence": 0.993, "span": {"offset": 412, "length": 10}}, {"content": "Allentown", "boundingBox": [236, 869, 339, 869, 339, 890, 236, 890], "confidence": 0.993, "span": {"offset": 423, "length": 9}}, {"content": "PA", "boundingBox": [347, 869, 370, 870, 370, 889, 346, 890], "confidence": 0.998, "span": {"offset": 433, "length": 2}}, {"content": "18109", "boundingBox": [387, 870, 442, 870, 441, 889, 387, 889], "confidence": 0.998, "span": {"offset": 436, "length": 5}}, {"content": "~", "boundingBox": [451, 870, 462, 870, 461, 889, 450, 889], "confidence": 0.826, "span": {"offset": 442, "length": 1}}, {"content": "610-433-6181x", "boundingBox": [470, 870, 621, 871, 620, 888, 469, 889], "confidence": 0.783, "span": {"offset": 444, "length": 13}}, {"content": "214", "boundingBox": [631, 871, 664, 871, 663, 887, 631, 887], "confidence": 0.996, "span": {"offset": 458, "length": 3}}], "lines": [{"content": "HAVEN HOUSE", "boundingBox": [225, 19, 460, 19, 460, 48, 225, 49], "spans": [{"offset": 0, "length": 11}]}, {"content": "RENEWING LIVES", "boundingBox": [654, 26, 653, 102, 640, 102, 640, 26], "spans": [{"offset": 12, "length": 14}]}, {"content": "Restoring Hope", "boundingBox": [560, 27, 644, 27, 644, 42, 560, 41], "spans": [{"offset": 27, "length": 14}]}, {"content": "SSOUDJRMY", "boundingBox": [577, 39, 576, 103, 560, 102, 560, 39], "spans": [{"offset": 42, "length": 9}]}, {"content": "SUMMER", "boundingBox": [27, 76, 338, 76, 338, 140, 27, 140], "spans": [{"offset": 52, "length": 6}]}, {"content": "G0 TEARS", "boundingBox": [584, 44, 644, 47, 644, 91, 581, 88], "spans": [{"offset": 59, "length": 8}]}, {"content": "Recovery Da", "boundingBox": [587, 88, 655, 91, 654, 104, 586, 100], "spans": [{"offset": 68, "length": 11}]}, {"content": "HAVED HOUSE", "boundingBox": [559, 104, 655, 105, 655, 121, 559, 120], "spans": [{"offset": 80, "length": 11}]}, {"content": "JOURNAL", "boundingBox": [163, 142, 626, 143, 625, 227, 163, 225], "spans": [{"offset": 92, "length": 7}]}, {"content": "DRIVE", "boundingBox": [206, 235, 648, 234, 648, 360, 206, 361], "spans": [{"offset": 100, "length": 5}]}, {"content": "Donated new journals are", "boundingBox": [366, 370, 642, 371, 641, 400, 366, 399], "spans": [{"offset": 106, "length": 24}]}, {"content": "given to individuals in therapy", "boundingBox": [365, 409, 685, 409, 685, 444, 365, 444], "spans": [{"offset": 131, "length": 31}]}, {"content": "to assist in the therapeutic", "boundingBox": [367, 449, 654, 449, 654, 481, 367, 479], "spans": [{"offset": 163, "length": 28}]}, {"content": "process", "boundingBox": [366, 495, 454, 493, 455, 517, 366, 520], "spans": [{"offset": 192, "length": 7}]}, {"content": "CONTACT", "boundingBox": [553, 504, 651, 504, 651, 525, 553, 525], "spans": [{"offset": 200, "length": 7}]}, {"content": "Lori Richardson", "boundingBox": [503, 531, 654, 532, 653, 557, 502, 555], "spans": [{"offset": 208, "length": 15}]}, {"content": "The healing", "boundingBox": [16, 575, 299, 576, 298, 629, 16, 626], "spans": [{"offset": 224, "length": 11}]}, {"content": "610-433-6181 x 214", "boundingBox": [503, 567, 681, 567, 681, 591, 503, 592], "spans": [{"offset": 236, "length": 18}]}, {"content": "Be brave", "boundingBox": [363, 593, 415, 594, 415, 606, 363, 606], "spans": [{"offset": 255, "length": 8}]}, {"content": "Be bold", "boundingBox": [364, 606, 419, 607, 419, 620, 364, 619], "spans": [{"offset": 264, "length": 7}]}, {"content": "or Irichardson@", "boundingBox": [503, 603, 664, 603, 664, 629, 503, 629], "spans": [{"offset": 272, "length": 15}]}, {"content": "Be beautol", "boundingBox": [363, 618, 430, 618, 430, 633, 363, 632], "spans": [{"offset": 288, "length": 10}]}, {"content": "power of", "boundingBox": [19, 641, 247, 632, 249, 681, 19, 687], "spans": [{"offset": 299, "length": 8}]}, {"content": "haven-house.com", "boundingBox": [504, 640, 674, 641, 674, 665, 504, 664], "spans": [{"offset": 308, "length": 15}]}, {"content": "writing", "boundingBox": [19, 690, 202, 690, 201, 746, 18, 742], "spans": [{"offset": 324, "length": 7}]}, {"content": "Follow us on facebook for Haven House", "boundingBox": [12, 790, 448, 790, 448, 820, 12, 820], "spans": [{"offset": 332, "length": 37}]}, {"content": "& mental health awareness info", "boundingBox": [14, 833, 380, 833, 380, 862, 14, 862], "spans": [{"offset": 370, "length": 30}]}, {"content": "A411 Union Boulevard, Allentown PA 18109 ~ 610-433-6181x 214", "boundingBox": [7, 869, 668, 869, 668, 889, 7, 890], "spans": [{"offset": 401, "length": 60}]}], "spans": [{"offset": 0, "length": 461}]}], "styles": []}, "metadata": {"height": 900, "width": 695, "format": "Jpeg"}}
    '''
    ocr_result = data['read']['content']
    if len(ocr_result) < 1:
        return ""
    string = f"Here is an image <|img_{r}|>.\n"
    string += f"Instruction: What are the texts in this image?.\nOutput: Doing OCR...\nThe texts are: {ocr_result}\n"
    return string


def seg_to_string(seg):
    seg = seg[0]
    sss = []
    for ss in seg:
        sss.append(round(ss, 2))
    return str(sss)


def get_point(seg):
    def is_point_in_polygon(polygon, point):  
        # Create a Path object from the polygon  
        poly_path = path.Path(polygon)  
  
        # Check if the point is in the polygon  
        return poly_path.contains_point(point)  
  
    # Your list of normalized floating-point values  
    segmentation_polygon =  seg[0]
    try:
  
        # Reshape the list into a 2D array with pairs of (x, y) coordinates  
        polygon = [(segmentation_polygon[i], segmentation_polygon[i + 1]) for i in range(0, len(segmentation_polygon), 2)]  
        count = 0
        while count < 10:
            point = (round(np.random.uniform(0, 1), 2), round(np.random.uniform(0, 1), 2))
            count = count + 1
            if is_point_in_polygon(polygon, point):
                return point

    except Exception as e:
        return None
    return None


def process_segment_caption(data, r):
    '''
        {"grounding": {"caption": "The image shows an adult businessman in a suit and tie, pointing to the clock with a look of anticipation as he arrives late for a deadline. The background is dark, emphasizing the businessman's urgency.", "restored_caption": "the image shows an adult businessman in a suit and tie, pointing to the clock with a look of anticipation as he arrives late for a deadline. the background is dark, emphasizing the businessman ' s urgency.", "phrases": ["an adult businessman", "a suit", "tie", "the clock"], "phrases_span": [[16, 36], [40, 46], [51, 54], [68, 77]], "boxes": [[0.20880848169326782, 0.0599251389503479, 0.9736554026603699, 0.9524948000907898], [0.20909810066223145, 0.27462637424468994, 0.97165846824646, 0.9534662961959839], [0.43140846490859985, 0.3443661630153656, 0.5882756114006042, 0.8048264980316162], [0.4611770510673523, 0.344607412815094, 0.5877744555473328, 0.5645797252655029], [0.4720001220703125, 0.6145464181900024, 0.553266167640686, 0.6625568866729736]], "phrases_to_boxes": [[0], [1], [2, 3], [4]], "boxes_to_phrases": [[0], [1], [2], [2], [3]], "phrase_box_index_to_score": [[[0, 0], 0.54], [[1, 1], 0.57], [[2, 2], 0.46], [[2, 3], 0.28], [[3, 4], 0.27]], "bbox_format": "xyxy", "bbox_normalized": true, "segmentation": [[[0.58154296875, 0.0638427734375, 0.55126953125, 0.06597900390625, 0.54150390625, 0.068115234375, 0.5380859375, 0.0701904296875, 0.521484375, 0.0701904296875, 0.51513671875, 0.07232666015625, 0.50830078125, 0.068115234375, 0.5048828125, 0.07232666015625, 0.48828125, 0.08087158203125, 0.481689453125, 0.0850830078125, 0.478515625, 0.08935546875, 0.478515625, 0.0936279296875, 0.4716796875, 0.10211181640625, 0.465087890625, 0.10638427734375, 0.465087890625, 0.11065673828125, 0.448486328125, 0.1297607421875, 0.448486328125, 0.142578125, 0.445068359375, 0.151123046875, 0.44189453125, 0.1531982421875, 0.44189453125, 0.166015625, 0.445068359375, 0.1744384765625, 0.448486328125, 0.1787109375, 0.455078125, 0.1829833984375, 0.455078125, 0.2340087890625, 0.45849609375, 0.2425537109375, 0.4619140625, 0.2509765625, 0.465087890625, 0.257568359375, 0.468505859375, 0.265869140625, 0.4716796875, 0.2744140625, 0.47509765625, 0.276611328125, 0.47509765625, 0.282958984375, 0.478515625, 0.289306640625, 0.481689453125, 0.29150390625, 0.485107421875, 0.300048828125, 0.48828125, 0.306396484375, 0.485107421875, 0.310546875, 0.47509765625, 0.31494140625, 0.45849609375, 0.31494140625, 0.451904296875, 0.319091796875, 0.445068359375, 0.319091796875, 0.428466796875, 0.325439453125, 0.421875, 0.325439453125, 0.411865234375, 0.329833984375, 0.4052734375, 0.329833984375, 0.395263671875, 0.333984375, 0.388671875, 0.333984375, 0.382080078125, 0.33837890625, 0.3720703125, 0.342529296875, 0.35205078125, 0.342529296875, 0.33544921875, 0.35107421875, 0.328857421875, 0.355224609375, 0.31884765625, 0.3681640625, 0.315673828125, 0.37451171875, 0.312255859375, 0.380859375, 0.30908203125, 0.38720703125, 0.30908203125, 0.39990234375, 0.3056640625, 0.41064453125, 0.30224609375, 0.4169921875, 0.299072265625, 0.42138671875, 0.29248046875, 0.425537109375, 0.2890625, 0.4296875, 0.28564453125, 0.442626953125, 0.28564453125, 0.453125, 0.282470703125, 0.45947265625, 0.279052734375, 0.461669921875, 0.279052734375, 0.47021484375, 0.2724609375, 0.4765625, 0.2724609375, 0.48095703125, 0.275634765625, 0.5, 0.265869140625, 0.5126953125, 0.255859375, 0.51904296875, 0.25244140625, 0.525390625, 0.245849609375, 0.52978515625, 0.245849609375, 0.54248046875, 0.2425537109375, 0.55126953125, 0.2325439453125, 0.5576171875, 0.21923828125, 0.56396484375, 0.2159423828125, 0.56787109375, 0.2159423828125, 0.5810546875, 0.212646484375, 0.59130859375, 0.2093505859375, 0.59375, 0.2093505859375, 0.6171875, 0.212646484375, 0.62548828125, 0.2159423828125, 0.6318359375, 0.21923828125, 0.63623046875, 0.2259521484375, 0.640625, 0.2392578125, 0.64453125, 0.2491455078125, 0.64453125, 0.265869140625, 0.64697265625, 0.279052734375, 0.65087890625, 0.29248046875, 0.65966796875, 0.295654296875, 0.666015625, 0.295654296875, 0.67431640625, 0.295654296875, 0.69580078125, 0.30908203125, 0.69384765625, 0.315673828125, 0.69384765625, 0.322265625, 0.69775390625, 0.328857421875, 0.7041015625, 0.328857421875, 0.70654296875, 0.348876953125, 0.71923828125, 0.35205078125, 0.7236328125, 0.35205078125, 0.72998046875, 0.348876953125, 0.73828125, 0.345458984375, 0.74658203125, 0.34228515625, 0.75732421875, 0.3388671875, 0.76611328125, 0.33544921875, 0.7724609375, 0.332275390625, 0.7744140625, 0.332275390625, 0.78076171875, 0.328857421875, 0.79150390625, 0.32568359375, 0.7978515625, 0.322265625, 0.80615234375, 0.31884765625, 0.81494140625, 0.315673828125, 0.8212890625, 0.312255859375, 0.82763671875, 0.30908203125, 0.82958984375, 0.30908203125, 0.8359375, 0.3056640625, 0.8447265625, 0.30224609375, 0.85302734375, 0.299072265625, 0.859375, 0.295654296875, 0.86572265625, 0.29248046875, 0.8681640625, 0.2890625, 0.87451171875, 0.28564453125, 0.880859375, 0.282470703125, 0.88916015625, 0.279052734375, 0.89794921875, 0.275634765625, 0.904296875, 0.2724609375, 0.91259765625, 0.26904296875, 0.92333984375, 0.265869140625, 0.93212890625, 0.262451171875, 0.9423828125, 0.259033203125, 0.955078125, 0.315673828125, 0.95751953125, 0.328857421875, 0.95751953125, 0.3388671875, 0.95751953125, 0.348876953125, 0.95751953125, 0.415283203125, 0.95751953125, 0.428466796875, 0.955078125, 0.45849609375, 0.955078125, 0.53466796875, 0.955078125, 0.5615234375, 0.95751953125, 0.8837890625, 0.955078125, 0.8837890625, 0.94677734375, 0.88720703125, 0.93212890625, 0.89013671875, 0.9296875, 0.89013671875, 0.92529296875, 0.88720703125, 0.9169921875, 0.8837890625, 0.9150390625, 0.8837890625, 0.8916015625, 0.88037109375, 0.87255859375, 0.876953125, 0.8701171875, 0.876953125, 0.86572265625, 0.88037109375, 0.8447265625, 0.876953125, 0.83837890625, 0.87353515625, 0.8359375, 0.87353515625, 0.8193359375, 0.87060546875, 0.80419921875, 0.8671875, 0.79345703125, 0.86376953125, 0.78076171875, 0.8603515625, 0.77001953125, 0.85693359375, 0.76171875, 0.85400390625, 0.759765625, 0.8505859375, 0.7490234375, 0.8505859375, 0.73193359375, 0.85693359375, 0.7275390625, 0.8671875, 0.7255859375, 0.87353515625, 0.72119140625, 0.8837890625, 0.716796875, 0.8935546875, 0.71484375, 0.900390625, 0.71044921875, 0.90673828125, 0.7021484375, 0.90673828125, 0.69775390625, 0.91357421875, 0.69140625, 0.9169921875, 0.68505859375, 0.92041015625, 0.6767578125, 0.923828125, 0.66796875, 0.923828125, 0.6318359375, 0.9267578125, 0.62548828125, 0.93359375, 0.619140625, 0.93701171875, 0.61279296875, 0.9404296875, 0.6044921875, 0.9404296875, 0.58935546875, 0.9404296875, 0.5830078125, 0.9404296875, 0.57666015625, 0.93359375, 0.572265625, 0.93359375, 0.56591796875, 0.93701171875, 0.55517578125, 0.9404296875, 0.55322265625, 0.9404296875, 0.5361328125, 0.943359375, 0.51513671875, 0.95703125, 0.50830078125, 0.95703125, 0.50439453125, 0.9599609375, 0.4892578125, 0.95703125, 0.48095703125, 0.95361328125, 0.474365234375, 0.9501953125, 0.472412109375, 0.94677734375, 0.45751953125, 0.94677734375, 0.444580078125, 0.943359375, 0.43408203125, 0.9404296875, 0.431884765625, 0.9404296875, 0.412841796875, 0.93701171875, 0.39990234375, 0.93359375, 0.3916015625, 0.9267578125, 0.383056640625, 0.9169921875, 0.376708984375, 0.90380859375, 0.3701171875, 0.87353515625, 0.36181640625, 0.8671875, 0.357421875, 0.8603515625, 0.357421875, 0.81396484375, 0.342529296875, 0.80712890625, 0.33837890625, 0.79052734375, 0.333984375, 0.77734375, 0.32763671875, 0.77099609375, 0.323486328125, 0.7607421875, 0.319091796875, 0.7373046875, 0.312744140625, 0.720703125, 0.30224609375, 0.7041015625, 0.289306640625, 0.701171875, 0.282958984375, 0.677734375, 0.26806640625, 0.677734375, 0.2509765625, 0.6845703125, 0.246826171875, 0.6943359375, 0.2425537109375, 0.69775390625, 0.23828125, 0.69775390625, 0.2340087890625, 0.7041015625, 0.2276611328125, 0.70751953125, 0.2213134765625, 0.70751953125, 0.21484375, 0.7109375, 0.2064208984375, 0.7177734375, 0.2021484375, 0.720703125, 0.19580078125, 0.720703125, 0.1893310546875, 0.7275390625, 0.1829833984375, 0.7275390625, 0.157470703125, 0.72412109375, 0.14892578125, 0.71435546875, 0.1446533203125, 0.7109375, 0.140380859375, 0.70751953125, 0.134033203125, 0.70751953125, 0.1297607421875, 0.7041015625, 0.12548828125, 0.69775390625, 0.12127685546875, 0.69091796875, 0.11065673828125, 0.6845703125, 0.104248046875, 0.6708984375, 0.09576416015625, 0.64794921875, 0.08087158203125, 0.634765625, 0.074462890625, 0.6181640625, 0.0701904296875, 0.611328125, 0.06597900390625, 0.6015625, 0.0638427734375]], [], [[0.465087890625, 0.716796875, 0.45849609375, 0.72119140625, 0.45849609375, 0.73193359375, 0.455078125, 0.73828125, 0.451904296875, 0.74658203125, 0.448486328125, 0.7490234375, 0.445068359375, 0.75732421875, 0.445068359375, 0.78076171875, 0.448486328125, 0.78515625, 0.455078125, 0.79150390625, 0.468505859375, 0.79345703125, 0.468505859375, 0.76171875], [0.53466796875, 0.348876953125, 0.52490234375, 0.35107421875, 0.51806640625, 0.355224609375, 0.51171875, 0.36181640625, 0.50830078125, 0.365966796875, 0.50830078125, 0.372314453125, 0.50830078125, 0.38720703125, 0.51171875, 0.39990234375, 0.50830078125, 0.406494140625, 0.50146484375, 0.412841796875, 0.50146484375, 0.4169921875, 0.48828125, 0.431884765625, 0.485107421875, 0.438232421875, 0.481689453125, 0.444580078125, 0.478515625, 0.451171875, 0.47509765625, 0.468017578125, 0.47509765625, 0.54248046875, 0.481689453125, 0.546875, 0.49169921875, 0.55322265625, 0.50146484375, 0.55517578125, 0.51171875, 0.55322265625, 0.51171875, 0.548828125, 0.51513671875, 0.54052734375, 0.51806640625, 0.5341796875, 0.521484375, 0.52783203125, 0.52490234375, 0.525390625, 0.52490234375, 0.51904296875, 0.5283203125, 0.5107421875, 0.53173828125, 0.50439453125, 0.5380859375, 0.497802734375, 0.5380859375, 0.491455078125, 0.544921875, 0.485107421875, 0.54833984375, 0.4765625, 0.55126953125, 0.468017578125, 0.55126953125, 0.455322265625, 0.5546875, 0.44677734375, 0.55810546875, 0.444580078125, 0.5546875, 0.404296875, 0.5546875, 0.39794921875, 0.55810546875, 0.3916015625, 0.56494140625, 0.385009765625, 0.57470703125, 0.378662109375, 0.57470703125, 0.37451171875, 0.5712890625, 0.3681640625, 0.56494140625, 0.36181640625, 0.55810546875, 0.355224609375, 0.54833984375, 0.348876953125]], [[0.53173828125, 0.348876953125, 0.51513671875, 0.359619140625, 0.51171875, 0.36376953125, 0.51171875, 0.402099609375, 0.50830078125, 0.408447265625, 0.498291015625, 0.419189453125, 0.498291015625, 0.42333984375, 0.49169921875, 0.4296875, 0.49169921875, 0.43408203125, 0.485107421875, 0.438232421875, 0.481689453125, 0.44677734375, 0.478515625, 0.455322265625, 0.47509765625, 0.472412109375, 0.47509765625, 0.54052734375, 0.478515625, 0.544921875, 0.48828125, 0.55126953125, 0.50146484375, 0.55517578125, 0.51171875, 0.55322265625, 0.51171875, 0.548828125, 0.51513671875, 0.54052734375, 0.51806640625, 0.5341796875, 0.521484375, 0.52783203125, 0.52490234375, 0.525390625, 0.52490234375, 0.51904296875, 0.5283203125, 0.5107421875, 0.53173828125, 0.50439453125, 0.53466796875, 0.497802734375, 0.5380859375, 0.495849609375, 0.54150390625, 0.4892578125, 0.544921875, 0.48291015625, 0.54833984375, 0.4765625, 0.55126953125, 0.468017578125, 0.55126953125, 0.453125, 0.5546875, 0.444580078125, 0.55810546875, 0.442626953125, 0.5546875, 0.412841796875, 0.5546875, 0.3935546875, 0.55810546875, 0.389404296875, 0.5712890625, 0.380859375, 0.57470703125, 0.376708984375, 0.5712890625, 0.3701171875, 0.56787109375, 0.365966796875, 0.5615234375, 0.36181640625, 0.55810546875, 0.355224609375, 0.55126953125, 0.35107421875, 0.544921875, 0.348876953125]], [[0.51513671875, 0.61474609375, 0.50830078125, 0.619140625, 0.50146484375, 0.619140625, 0.49169921875, 0.62109375, 0.481689453125, 0.62744140625, 0.478515625, 0.63427734375, 0.478515625, 0.64697265625, 0.48828125, 0.6533203125, 0.498291015625, 0.65966796875, 0.54150390625, 0.6572265625, 0.544921875, 0.640625, 0.544921875, 0.6298828125, 0.53466796875, 0.62109375, 0.5283203125, 0.6171875]]], "segmentation_polygon_normalized": true}}
    '''

    segment = data['grounding']['segmentation']
    phrases = data['grounding']['phrases']
    boxes =  data['grounding']['boxes']
    phrases_to_boxes = data['grounding']['phrases_to_boxes']
    boxes_to_phrases = data['grounding']['boxes_to_phrases']
    string_list = []
    if len(phrases) < 1:
        return ""
    '''
    string = f"Here is an image <|img_{r}|>.\n"
    string += "Instruct: List the objects and their segmentations in this image.\nOutput: Sure, here are some of the objects and their segmentations:\n"
    count = 0
    for i, pp in enumerate(phrases):
        bxs = phrases_to_boxes[i]
        bx = []
        for bb in bxs:
            if len(segment[bb]) > 0:
                bx.append(seg_to_string(segment[bb]))
        if len(bx) < 1:
            continue
        bx = " and ".join(bx)
        string += f"{count}. Object name: {phrases[i]}, segmentation: {bx}.\n"
        count = count + 1
    string_list.append(string)
    '''
    string = f"Instruct: List all the objects in this image <|img_{r}|>.\n"
    lst = [cc for cc in phrases]
    string += f"Output: The objects are {', '.join(lst)}."
    string_list.append(string)
    
    #string = f"Instruct: What is in this image?\nOutput: {data['grounding']['caption']}"
    #string_list.append(string)
    #for i, bb in enumerate(boxes):
    
    for i, pp in enumerate(phrases):
        string = f"Here is an image <|img_{r}|>.\n"
        
        bxs = phrases_to_boxes[i]
        bx = []
        for bb in bxs:
            if len(segment[bb]) > 0:
                bx.append(seg_to_string(segment[bb]))
        if len(bx) < 1:
            continue
        bx = " and ".join(bx)

        string += f"Instruct: Tell me what is the segmentation of the object **{pp}** in this image.\nOutput: Sure, the segmentation is {bx}.\n"

        string_list.append(string)

    for i in range(len(segment)):
        string = f"Here is an image <|img_{r}|>.\n"
        if len(segment[i]) < 1:
            continue
        #print(segment[i])
        point = get_point(segment[i])
        if point is None:
            continue
        obj = phrases[boxes_to_phrases[i][0]]
        string += f"Instruct: Tell me what is the object name corresponding to the point {point}?\nOutput: The object name is {obj}.\n"
        
        
        string_list.append(string)

    return  "\n<|endoftext|>\n".join(string_list)

def process_uu_caption(data, r):
    #{"captions": ["3 bhk independent house for sale in kharghar"]}
    caption = data['captions']
    string_list = []
    for cc in caption:
        string = f"Here is an image <|img_{r}|>.\n"
        string += f"Instruction: What is in this image?\nOutput: {cc}\n"
    string_list.append(string)
    #string = f"Instruct: What is in this image <|img_{r}|>?\n"
    #string += f"Output: {caption}\n"
    return "\n<|endoftext|>\n".join(string_list)
    #string = f"Instruct: What is in this image <|img_{r}|>?\n"
    #string += f"Output: {caption}\n"
    #return string


class florence_random_sampler:

    def __init__(self, process_root = "/mnt/phyagieast2pr/datasets/vision_datasets/florence/",  root = "/mnt/phyagieast2pr/datasets/vision_datasets/florence/florence_root/", json_path = "", process_func = None, version_number = 0, max_len = 4096):
        #process_root: root of the blob storage
        #root: root of the json file
        self.text_chunk_size_dict = {}
        self.text_to_image_dict  = {}
        self.text_to_process_root_dict  = {}
        self.process_func = process_func
        self.version_number = version_number
        self.max_length = max_len
        try:
            with open(os.path.join(root, json_path)) as f:
                h = json.load(f)
                data_sources = h['data_sources']
                for dd in data_sources:
                    pr = os.path.join(process_root, dd['container_name'])
                    img_path = dd['image_blob_template']             
                    text_path = dd['text_blob_template']
                    total_chunks = dd['total_chunks']
                    
                    '''
                    #update the maximum chunks, for alignment purpose.
                    if img_path in self.image_chunk_dict:
                        self.image_chunk_dict[img_path] = max(total_chunks, int(self.image_chunk_dict[img_path]))
                    else:
                        self.image_chunk_dict[img_path] = int(total_chunks)
                    '''
                    self.text_chunk_size_dict[text_path] = int(total_chunks)
                    #update the maximum chunks, for alignment purpose.
                    self.text_to_image_dict[text_path] = img_path
                    self.text_to_process_root_dict[text_path] = pr
                    print(text_path, img_path, total_chunks, pr)
            self.initialized = True
        except Exception as e:
            self.initialized = False
            print(e)
            pass
        if len(self.text_chunk_size_dict) < 1:
            self.initialized = False

    def random_sample(self):
        #import numpy as np
        import random  
        import numpy as np
        import os
        import json
 
        # Convert the dictionary keys to a list and randomly sample one  
        text_path = random.choice(list(self.text_to_image_dict.keys())) 
        image_path = self.text_to_image_dict[text_path]
        chunk_id = np.random.randint(0, self.text_chunk_size_dict[text_path])
        root = self.text_to_process_root_dict[text_path]
        text_file_name = os.path.join(root, text_path).format(chunk_id)
        image_file_name = os.path.join(root, image_path).format(chunk_id)
        with open(text_file_name,'r', encoding = 'utf-8') as f:
            h = f.readlines()
        cc = []
        #over sample, and then shuffle.. can be adjusted for faster speed
        total_length = 0
        while total_length < self.max_length * 100:
            idx = np.random.randint(len(h))
            entry = h[idx]
            
            key, value = entry.split("\t")
            dict = json.loads(value.strip())
            try:
                cc_tmp = self.process_func(dict, idx + 1)
            except Exception as e:
                cc_tmp = ""
                '''
                print("bug" + "*" * 100, flush = True)
                print(e)
                print(h[0])
                print(h[idx])
                print("bug file name" + "-" * 100, flush = True)
                print(text_file_name, dict)
                '''
            total_length = total_length + len(cc_tmp)
            cc_tmp = cc_tmp.split("\n<|endoftext|>\n")
            cc.extend(cc_tmp)
        import random
        random.shuffle(cc)
        cc = "\n<|endoftext|>\n".join(cc)
        idx = np.random.randint(0, len(cc) - 10 * self.max_length)
        cc = cc[idx:idx + 10 * self.max_length]

        return cc, image_file_name
        

if True:
    class FlorenceSampler:
        def __init__(
            self, 
            process_root = "/mnt/phyagieast2pr/datasets/vision_datasets/florence/",  
            root = "/mnt/phyagieast2pr/datasets/vision_datasets/florence/florence_root/",  
            image_processor = None, 
            vision_tower = None, 
            max_len = 4096, 
            max_offset = 1e9, 
            tokenizer = None, 
            to_clip = 0, 
            tokenizer_option = 1, 
            shift_labels=False):

            self.process_root = process_root
            self.root = root 

            try:
                self.max_len = args.max_len
            except Exception as e:
                self.max_len = max_len

            try:
                self.max_offset = args.max_offset
            except Exception as e:
                self.max_offset = max_offset
                

            try:
                self.to_clip = args.to_clip
            except Exception as e:
                self.to_clip = to_clip
            
            self.label_offset = 1 if shift_labels else 0
        
            #tokenization function takes input a string, and return a list of numbers.
            self.tokenizer = self.tokenize_chunk_image
            
        
            if image_processor is None:
                self.image_processor = self.load_image_processor()
            else:
                self.image_processor = image_processor
        
            #can be removed
            if vision_tower is None and to_clip == 1:
                self.vision_tower = self.load_embedding()
            else:
                self.vision_tower = vision_tower
            if tokenizer is None:
            
                if tokenizer_option == 1:
                    from transformers import AutoTokenizer
                    from transformers import AddedToken  
                    import torch  
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",trust_remote_code=True)
                    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
                    tokenizer.add_tokens(AddedToken("</s>",normalized=False, rstrip=True, lstrip=False))
                    special_tokens = ["<|end|>", "<|system|>", "<|tag|>", "<|user|>", "<|assistant|>", "<|continue|>", "<|function_list|>", "<|function_call|>", "<|function_output|>", "<|step|>", "<|raw|>"]
                    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken(special_token, normalized=False, rstrip=True, lstrip=False) for special_token in special_tokens]})
                    #args.old_tokenizer =  AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
                    print("loading new llama tokenizer...", flush=True)
                    #count = count + 1
                    self.tokenizer_temp = tokenizer
                #can be replaced
                else:
                    self.tokenizer_temp = None
            else:
                self.tokenizer_temp = tokenizer
            
                        
            self.sampler_list = []
            vn = 0
            for rc in [ "openimg", "O365", "in22k", "CCS", 'laionraw']:
                
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alt2nar.json", max_len = self.max_len, process_func = process_regular_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alt2nar-gdino.json", max_len = self.max_len, process_func = process_grouding_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alt2nar-gdino-wpolygon.json", max_len = self.max_len, process_func = process_segment_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alttext.json", max_len = self.max_len, process_func = process_text_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alttext-gdino.json", max_len = self.max_len, process_func = process_grouding_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/alttext-gdino-wpolygon.json", max_len = self.max_len, process_func = process_segment_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/best_query.json", max_len = self.max_len, process_func = process_dense_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/gptv-50k-ft.json", max_len = self.max_len, process_func = process_gptv_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/gptv-50k-ft-gdino.json", max_len = self.max_len, process_func = process_grouding_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/gptv-50k-ft-gdino-wpolygon.json", max_len = self.max_len, process_func = process_segment_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/ocr.json", max_len = self.max_len, process_func = process_ocr_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/od.json", max_len = self.max_len, process_func = process_od_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/region_proposal.json", max_len = self.max_len, process_func = process_od_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/uucap.json", max_len = self.max_len, process_func = process_uu_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/uucap-gdino.json", max_len = self.max_len, process_func = process_grouding_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)
                ts = florence_random_sampler(process_root = process_root, root=root, json_path = f"{rc}/uucap-gdino-wpolygon.json", max_len = self.max_len, process_func = process_segment_caption,  version_number = vn)
                if ts.initialized:
                    self.sampler_list.append(ts)

            self.special_string = "5523c6a478d00c6882136b6b2da90fc0991a4cd36b905c7351f669e3e3a8d75dfb280b68de2c803529d1acda14d42f885c75d54441dcfc06f6e529d0b7952af7b5a82cda0b3b927660997c383315bbf5813ff1a564c32f5219e8f2415a37bb2a164b88b1274316e885dbd159823d25857233c4588995d3664515ef517fd9814a94aaaaf55f534e0474f1d99088a6afc39d223cbfac14ccdb161510910a61c9bf4b8c7486dc397d80f5da6298f178d5b4cc31b6dc46cfcd1499ab200591b00e6a36f31f0d580816a46fdbb6de8f522b1dbeec9b25a15f38772a4a5bae39a5445299d0c8b8512dd1afb43c341117086411c63c8ec2f2c12336eaa283f6f3a94dcad89e64c27a09677c90135d1b8305c44f71045f14126872cd90fb0638bdc9d754a7594a61c2007727d922a8269e1d2ec09bcd6a6802e2f610b64bb0b913915784059794834bb349bacc3774791fb11de926d4d17b7a85972c0cbdaa4ee0952edbead335942614e2a9d24b9e1cd796f4430aca310a9ae52db7e9ba8eebef209469c3cc401d0b111ceca7da97ac272677b37eb152beb8c934a7f2835afe637be2709312b967e2cf04cda185bcf2b6d7b017baad3cfa8bf2655471dea5b8c0855c0da4b2ae1efacb448eaeed15968ff126eb5b6adf2347fa3d831758437cce13a02c8379867a2ea367661ed270805cb9cdf817cd12b3589833dff12886fa3e5e3814ace4797c8689a4cba4fa065d9399bbde23c98fede06425c55e0d560c43e68ea0486a78b99178859414ab2b84427d1d8d34a4a52870eb1b79693311fb26770df8e85824db84bfb27fa5539f987af544a5f3c19296287207436b887cf8735d9964302d9cf123369dd2f725374ed7f13d4ca0407e1e79a1ce08e44159b635e92ebb948c3f0e62840dc075864c572b09e897922ce7ffdd5df3b8750e3d1ce828dc4349b729a5c47fd4944fec2e55a04a4237b793c34b2bec8804c59520cf998492aee5b5fbeb54ec2b78794a127a7fc6476c5ec3d8147d48c38470efd021ab2c8ca83282b259f0bc1327890b2cd9cf4599b97f0f08adc7ed45269cbd403fe1e7943c8f7ea033281c24ffa21c684a4c034afb2bb28ac66c016b9623da2245444d28afeb07e4a14bae0a02709b3be796d6351d620044faa38333b09a96062018b1ea5260f8bb96be36a88aca56184d3eb76f425bde7eedbf3e63b07eeff67cce9f0b1382a9bde08c9d88470440519e6094de8c21b1edbcdd77a0706e6dc582d8835f68731b8d156790681019658176864bb59cf27b2873a15c2d36ece5236433d8bc81a04243a39af6edb249e4d7e0910d54949825fd029383e717efbec65abd9ea95574ddb7b9c454e0a74ed5c041e7bbe9958392987b459750e239b6d5dfda12c38ff75ac99b7323edecbbb8e473b994712b8ca4598cd17ad03316cdc1024ac6338538d258f01f699b3301c9a0457439b404b823cee4b0185cfffc99d2d5575df174a4a96b184c7f5bbdd1f33360d524b4deaa13d4dc68f4d0c910797b41fe88d27dbd6b0f2af3c1e299e350cdfa11aed37b94bee1a91d3084bc423e321ec2dad9f92a362b68368c67be280f218808aef74b135269ec657d3729ed02fe3c0c583f1016f5b3751e32d02858c6039572dbe426b0d40964c009c730109a1850fd63670a8700cd65f5eeceedc352ba0611077f8191ab6dd86f7c5f4cfeddc5d28865b61a4d32e23e52166ad6019ea5dc87c036fbdbf8f62b26643f3707f893be15429cd60457c78ec2ed8c69bcbb2b108ab99b06d082983379e80323d9f854e2b1c3eced4b85a0bb621927bd75728ad36bf55eebe7ef9c591f57b44b34d0d5ce788f5a57ca9537c97747fdb11fce249ac22b126b3085f810cc9a343677b737288a7370c87289c195c1318dd1a8ad1fe93f70665f9b0486091ec5112d5984d8567b114395a6774dd854bf46b61fa911afa29ea050a605efd89aa4e29243b9d2b07fc9c293be5d1855bc7417db515b862912a3f5459531ed16f47cd293809a19d4654c75801db867e7e742dc54cfe6662f7f28fdf5d477eb1d1374e5801531875b9e82c3fcf6500764362d4c8f5eea7965603fb2369f007a541b8e3b07e7cadf05b396ba6ecc9ad09ec2e5a5121eec4d1edb487a1fcd210171b6a9d7d1aea436066993f5ed3cd257db168ab2c70546770314e2d228517d8d82aaf663cff1a15beca96dbef1e301c7ac601649b9ca8cbd2e4c6c6f8a2383f536d12dbb3502e740607a60509669b1287b7a7b94df6780486d16ba6a5c70b4bbc727bbe30e3a9538399aa9b450328cabfb3b11d7764145cf8fe5b9b4fe3aed12aad6833468004713a9d12eae9d27d044cbaec8f719ede5e6851adbd19e6533b2ec3ccbb57d1d70143011ce58526407ad7e1b868db90184a740db73dd94015b5d8f29647cfdb3c6dcf7b10b72faf3c5e4043282b27765399937557f1874927393465dbf5ef2d2f32bcbbb935ff9ce954366a3bba07d5fe876f88f637cd732b7fcef89373cfbac62d87a8f2691b8154c5fc6373f2d2630f888542a3ee9af3ff694bb3b98af2159a5d8345cff33f31fa346b114710270bcc43898af611b5b5f00e86a2d83507026bcd77524995e25896b8c5f64e759dd76f54c71bc0062e0b7be5d20a33b62cf3d0e2343f3199f3f4415e0904c24d7f42efb04ddf67e1a8316c3540c367ee033b16cf7f655af1d399e259ec13c753e92921a8c82"
        def get_size(self):
            #dummy
            return 10
            #import os
            #assuming 3 chars per token..very rough estimate
            #return os.path.getsize(self.text_file_name)//3
        def find_and_replace_img2(self, string):
          # Initialize an empty list to store the integers X
          img_nums = []

          # Initialize a variable to store the modified string
          new_string = ""

          # Initialize a variable to track the start index of a potential <|img_X|> substring
          start = -1

          # Loop through the characters of the string
          for i, c in enumerate(string):
            # If the character is < and the start index is -1, mark the start index as i
            if c == "<" and start == -1:
              start = i
    
            # If the character is > and the start index is not -1, check if the substring between start and i is a valid <|img_X|>
            elif (c == ">" and start != -1) or i == len(string) - 1:
              # Extract the substring between start and i
              sub = string[start:i+1]

              # Check if the substring matches the pattern <|img_X|>, where X is a multi-digit integer
              if sub.startswith("<|img_") and sub.endswith("|>") and sub[6:-2].isdigit() and len(sub[6:-2]) >= 1:
                img_idid = int(sub[6:-2])
                # If yes, append the integer X to the img_nums list
                img_nums.append(int(sub[6:-2]))
                #print(int(sub[6:-2]))

                # Replace the substring with 32 # characters in the new_string
                new_string += self.special_string
    
      
              # Otherwise, append the substring as it is to the new_string
              else:
                new_string += sub
      
              # Reset the start index to -1
              start = -1
    
            # If the character is not < or > and the start index is -1, append the character as it is to the new_string
            elif start == -1:
              new_string += c
  
          # Return the new_string and the img_nums list
          return new_string, img_nums

        def tokenize_chunk_image(self, subchunk):
      
            subchunk, img_nums = self.find_and_replace_img2(subchunk)
            subchunks = subchunk.split(self.special_string)
            tt = []

            for ct, ss in enumerate(subchunks):
                if len(ss) >= 1:
                    input_ids2 = self.tokenizer_temp.encode(ss)
                    tt = tt + input_ids2
                try:
                    tt.append(- img_nums[ct])
                except Exception as e:
                    pass
                    #print("error:", e)
                    #print(len(img_nums), ct)
            
            return tt


        def load_image_processor(self, vision_tower_name = "openai/clip-vit-large-patch14"):
            """loading the CLIP vision model part and keep the backbone frozen"""
            # vision_tower_name = "openai/clip-vit-large-patch14"
            from transformers import CLIPVisionModel, CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
   
            return image_processor


        def load_and_process_image(self, binary, image_processor, size=[512, 512], center_crop=False, padding = False):
            import numpy as np
            from PIL import Image, ImageOps
            import io
            import torch
 
            image = Image.open(io.BytesIO(binary)).convert("RGB")
            from PIL import Image  
            import matplotlib.pyplot as plt  
            # Display the image  
    
            if center_crop:
                image = image.crop(
                    (
                        (image.width - min(image.width, image.height)) // 2,
                        (image.height - min(image.width, image.height)) // 2,
                        (image.width + min(image.width, image.height)) // 2,
                        (image.height + min(image.width, image.height)) // 2,
                    )
                )
 
            if size is not None and padding:
                # Calculate the ratio and resize
                ratio = min(size[0] / image.width, size[1] / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size)
 
                # Calculate padding
                delta_width = size[0] - new_size[0]
                delta_height = size[1] - new_size[1]
                padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        
                # Pad the resized image
                image = ImageOps.expand(image, padding)
            elif size is not None:
                image = image.resize((size[0], size[1]))

        
            try:
        
                img_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            except Exception as e:
                print(e)
                img_tensor = torch.zeros([3, 224, 224])
        
            return img_tensor


        def decode_binary_example(self, filename: str, j: int) -> bytes:
            """
            Args:
                filename (str): path to the binary chunk file
                j (int): index of the example to retrieve
            Returns:
                A bytes object representing the j-th example from the binary chunk.
            """
            endianness = "little"
            index_byte_len = 8  # 64 bit indices

            with open(filename, 'rb') as f:
                # Seek to the j-th offset value and read it
                f.seek(j * index_byte_len)
                offset_j = int.from_bytes(f.read(index_byte_len), byteorder=endianness)

                # Seek to the (j + 1)-th offset value and read it
                f.seek((j + 1) * index_byte_len)
                offset_j1 = int.from_bytes(f.read(index_byte_len), byteorder=endianness)

                # Seek to the j-th example and read it
                f.seek(offset_j)
                example_j = f.read(offset_j1 - offset_j)

            return example_j
        
        def get_image_embedding(self, imgs):

            vision_tower = self.vision_tower
            import torch
            device =  "cpu"
            # device = 'cpu'
            dtype = torch.float16
            select_layer = -2  # need to update
            select_feature = "patch"  # need to update
            with torch.no_grad():
                imgs = imgs.to(device, dtype=dtype)
                image_forward_out = vision_tower(imgs, output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[select_layer]
                if select_feature == "patch":
                    image_feature = image_feature[:, 1:]
                elif select_feature == "cls_patch":
                    image_feature = image_feature
                else:
                    raise NotImplementedError
            return image_feature.cpu().numpy()
        

        def load_embedding(self, vision_tower_name = "openai/clip-vit-large-patch14"):
            """loading the CLIP vision model part and keep the backbone frozen"""
            # vision_tower_name = "openai/clip-vit-large-patch14"
            from transformers import CLIPVisionModel, CLIPImageProcessor
            import torch
            image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
            vision_tower.requires_grad_(False)
            vision_tower = vision_tower.to('cpu')
            #vision_tower = torch.nn.DataParallel(vision_tower)
            return vision_tower

        
        def sample(self, num_chars = 0):
            import numpy as np
            import os
            import time  
            if num_chars == 0:
                num_chars = self.max_len * 10
            
            
            # Record the start time  
            i = np.random.randint(0, len(self.sampler_list))
            sampler = self.sampler_list[i]
            chars, image_dir = sampler.random_sample()
            

            #print(chars)
            tokenized_array = self.tokenizer(chars)
            if len(tokenized_array) > self.max_len:
                import numpy as np
                idx = np.random.randint(0, len(tokenized_array) - self.max_len)
                tokenized_array = tokenized_array[idx:idx + self.max_len]
            if len(tokenized_array) <= self.max_len:
                tokenized_array = tokenized_array + [0 for rr in range(self.max_len - len(tokenized_array))]
   
        
            images = []
            tkr_new = []
            for k, j in enumerate(tokenized_array):
                if j < 0 and len(tkr_new) < self.max_len - 257:
                    
                    try:
                        #print(image_name, image_id)
                        if True:
                            import time
                            start_time = time.time() 
                            binary = self.decode_binary_example(image_dir, abs(j) - 1)
                        
                            img = self.load_and_process_image(binary, self.image_processor)
                            #print(img.shape)
                            if self.to_clip == 1:
                                images.append(self.get_image_embedding(img.unsqueeze(0)))
                            else:
                                images.append(img)
                            # Record the end time  
                            end_time = time.time()  
                            
                            # Calculate and print the running time  
                            running_time = end_time - start_time  
                            if np.random.randint(0, 1000) == 0:
                                print("Running time of processing image: ", running_time, "seconds", flush = True)
                        tkr_new = tkr_new + [j] * 256  
                    except Exception as e:
                        #some image can not be decoded etc.
                        print(e)
                        pass
                      
                elif j < 0:
                    tkr_new = tkr_new + [-int(self.max_offset) - 1] * 256
                else:
                    tkr_new.append(j)
                if len(tkr_new) > self.max_len + 1:
                    break
                
            if len(tkr_new) < self.max_len:
                tkr_new = tkr_new + [0 for i in range(self.max_len - len(tkr_new) + 1)]
            
            labels = tkr_new[0+self.label_offset:self.max_len+self.label_offset]
            tkr_new = tkr_new[0:self.max_len]
            
            #print(f"collecting image finished length: {len(images)}", flush = True)
            import numpy as np
            import torch
            #print(images)
            if len(images) == 0:
                return torch.tensor(tkr_new).long(), torch.tensor([]), torch.tensor(labels).long()
            
            
            if self.to_clip == 1:
                images =  np.concatenate(images)
                return torch.tensor(tkr_new).long(), images.reshape(-1, images.shape[-1]), torch.tensor(labels).long()
            else:
                return torch.tensor(tkr_new).long(), torch.stack(images), torch.tensor(labels).long()



if __name__ == '__main__':
    FS = FlorenceSampler()
    for i in range(100):
        t, img, labels = FS.sample() 
        import pdb 
        pdb.set_trace()


