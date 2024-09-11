import google.generativeai as genai
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import cv2
import os
import shutil
model2=genai.GenerativeModel('gemini-1.5-flash')
genai.configure(api_key='AIzaSyAo5RwHPGsmlRiVjCE0o-ELD-DDn6Kg8mY')


model=genai.GenerativeModel('gemini-pro')

# path_video = 'input_video/input_video_2.mp4'
# path_folder='output_frames'

#Generating Frames

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    frame_number = 0

    while success:
        # Save every frame as a .jpg file
        img_filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
        cv2.imwrite(img_filename, frame)
        # print(f"Saved: {img_filename}")

        # Read the next frame
        success, frame = video.read()
        frame_number += 1

    video.release()
    # print(f"Extraction complete. Saved {frame_number} frames in {output_folder}.")
    return frame_number

#Generating Comments

def generate_output(path_folder, path_video):
  commentary = []
  # response1 =""
  strtr = "" #Give the commentary of the following in a sentence in 10 words only:\n
  prompt ='you are a commentator for the game going on, please give the commentary on it and give more emphasis on the events going on such as type of shots being played, player actions, scoring etc in strictly atmost 5 words of one sentence only:\n'
  frame_num = extract_frames(path_video, path_folder)
  for i in range(0,frame_num,20):
    path_image = f'output_frames/frame_{i}.jpg'
    img = PIL.Image.open(path_image)
    if(i%60==0):
      response1 = model2.generate_content(img)
      strtr = strtr +" "+ response1.text
      response2 = model.generate_content(prompt+strtr)
      print(response2.text)
      commentary.append(response2.text)
      strtr = ""
    else:
      response1 = model2.generate_content(img)
      strtr = strtr +" "+ response1.text
  if(frame_num%60!=0):
      response2 = model.generate_content(prompt+strtr)
      print(response2.text)
      commentary.append(response2.text)
  shutil.rmtree(path_folder)
  return commentary, frame_num