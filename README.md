# Game Sense By Team Brahma_Hackers

## About

## Project Working Video
https://github.com/user-attachments/assets/7568c885-078b-4c49-9f89-de16ada76ed0

https://github.com/user-attachments/assets/43d134e9-50d5-409e-8952-f12802ad8099

## Pre-trained model Weights
[Link to Download](https://drive.google.com/drive/folders/1lvTdDgxicRz09A_YLwlP9dK4daMWgIU4?usp=sharing) 

## How to get started 🚀
First, we need to have all the requirements before hosting into our local system.
### Requirements:
+  git installed in your system
+  python environment setup (v3.0 or more preferred)
### Steps to Follow:
+ Clone the Repository
```bash
git clone https://github.com/anishhello/gamesense_adobe_gensolve.git
```
+ Open the terminal in the folder where all these files are saved
+ Then in cmd
```bash
pip install -r requirements.txt
```
+ Replace your Gemini-API Keys in commentary.py, AI_Insights.py and PlayerMovementMetric.py
+ Download the pre-trained weights from the drive link provided
+ Use the command to run the app
```bash
streamlit run app.py
```
+ Upload the Video and Download it to view the results. 🤩

## Limitations and Restrictions:

+ Make sure you have a stable internet connection. It is required to generate content from GenAI.
+ Our Project expects 1280x720 resolution video(30fps). (To overcome this problem we integrated a resizer, giving any resolution that converts it to 1280x720 itself.)
+ Due to CPU Processing on the local system, some of the AI Generation results may show response time exceeded or quota exhausted. So use Colab or Kaggle to compute the real-time AI generation.
+ Moreover, the CPU isn't able to process a video of more than 15 seconds so, divide the bigger video into smaller ones and obtain the results.








