# GameSense: AI-Powered Insights for Tennis 🎾

### Developed by: Arin Gupta, Anish Kumar, Abhishekh Sinha (Team Brahma_Hackers)
#### This is the final Project for the National Finale of Adobe GenSolve 2024.

## Introduction

**GameSense** is a computer vision and AI/ML-driven tool that provides automated insights for two-player sports like tennis, badminton, and table tennis. The project focuses on player tracking, event detection, score-keeping, and real-time metrics. A stretch goal includes developing real-time AI-generated commentary.

## Key Features
- **Player Tracking**: Detects player positions and movements on the court.
- **Event Detection**: Identifies key events like hits, bounces, and fouls.
- **Score-Keeping**: Automates the scoring process based on event detection.
- **Real-Time Metrics**: Displays player statistics like distance covered, speed, and shot accuracy.
- **AI Commentary (Future Work)**: Offers live AI commentary on gameplay.

## Approach

1. **Game Selection**: Focused on tennis for initial development.
2. **Model Research**: 
   - **Faster R-CNN with ResNet-50** for player and court detection, providing better accuracy in cluttered scenes.
   - **TrackNet** for ball tracking and bounce detection, offering superior temporal consistency for tracking fast-moving balls.
   - **CatBoostRegressor** for ball bounce detection.
3. **Data Handling**: Uses annotated datasets from Roboflow for training. Handles missing ball tracking data through interpolation techniques.
4. **Processing Information**: 
   - Automated scoring based on hit and bounce detection.
   - Logic development for rally analysis using ball tracking.
   - Calculation of player metrics such as speed and distance covered.
5. **Real-Time AI Generation (Stretch Goal)**: Fetches real-time AI services via Gemini API for gameplay insights and commentary.

## Methodology
- **Player & Court Detection**: Leverages Faster R-CNN with ResNet-50 to ensure consistent accuracy across frames.
- **Ball Detection**: TrackNet provides accurate tracking of the ball even in complex scenes.
- **Event Detection**: Hits, bounces, and out-of-play events are detected through a combination of Y-coordinate processing and interpolation.
- **Metric Calculation**: Computes player movement metrics (distance, speed) using tracked coordinates.

## Results
- **Player Detection**: Accurate detection of player positions on the mini-map.
- **Ball & Bounce Detection**: High accuracy in ball tracking and bounce detection.
- **Score & Event Detection**: Automated scoring and event logging with rally analysis.
- **Metrics Visualization**: Real-time display of player performance metrics.
- **Streamlit Integration**: A Streamlit app interface for video uploads, processing, and CSV generation of player metrics.

## Improvements
- **Handling Inaccuracies**: Interpolation of missing ball/player detection and improvements in dataset quality.
- **Model Improvements**: Increasing model complexity to enhance detection accuracy.
- **Scoring & Event Detection**: Further refinements in rally and event detection logic.

## Future Prospects
1. **Player Performance**: Real-time feedback on movements and shot accuracy for training and injury prevention.
2. **Coaching & Strategy**: AI insights help coaches personalize training and develop better game strategies.
3. **Fan Engagement & Officiating**: Enhancing broadcasts with interactive data, improving officiating accuracy with AI-driven analysis.


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
+ Due to CPU processing on the local system, some AI generation results may show that the response time was exceeded or the quota was exhausted. So use Colab or Kaggle to compute the real-time AI generation.
+ Moreover, the CPU isn't able to process a video of more than 15 seconds so, divide the bigger video into smaller ones and obtain the results.








