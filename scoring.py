import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from tracking_utils import calculate_speed, angle_and_quad, calculate_distance   #, track_player_speed_and_direction
from keypointdiff_and_scalefactor import keypoint_differences_in_metres, scaleFactor, centre_court
from ball_analysis import get_ball_shot_frames
from utils import scene_detect
import math
import argparse
import torch

def scores(homography_matrices, kps_court, frame_nums, ball_track, df_track_ball, bounces):
    l=[]
    for i in range(frame_nums):
        if kps_court[i] is not None: 
            pts=[]
            p5=kps_court[i][4]
            p6=kps_court[i][5]
            p7=kps_court[i][6]
            p8=kps_court[i][7]
            p9=kps_court[i][8]
            p10=kps_court[i][9]
            p11 = kps_court[i][10]
            p12 = kps_court[i][11]
        
            p5ref=cv2.perspectiveTransform(p5.reshape(1,1,2),homography_matrices[i])
            p6ref=cv2.perspectiveTransform(p6.reshape(1,1,2),homography_matrices[i])
            p7ref=cv2.perspectiveTransform(p7.reshape(1,1,2),homography_matrices[i])
            p8ref=cv2.perspectiveTransform(p8.reshape(1,1,2),homography_matrices[i])
            p9ref=cv2.perspectiveTransform(p9.reshape(1,1,2),homography_matrices[i])
            p10ref=cv2.perspectiveTransform(p10.reshape(1,1,2),homography_matrices[i])
            p11ref=cv2.perspectiveTransform(p11.reshape(1,1,2),homography_matrices[i])
            p12ref=cv2.perspectiveTransform(p12.reshape(1,1,2),homography_matrices[i])
            pts.append(p5ref)
            pts.append(p6ref)
            pts.append(p7ref)
            pts.append(p8ref)
            pts.append(p9ref)
            pts.append(p10ref)
            pts.append(p11ref)
            pts.append(p12ref)
            l.append(pts)
        else:
            pts=[]
            for j in range(8):
                pts.append(np.array([[[np.nan, np.nan]]], dtype=np.float32))
            l.append(pts)    
    
    
    for i in range(frame_nums):
        if not math.isnan(ball_track[i][0]):
            frame_wanted = i
            break
    
    score = {
    '1':[0],
    '2':[0],
    'Event':['No Detection'],
    'Rally_Break':[0]
    }
    
    for i in range(frame_wanted):
        score['1'].append(0)
        score['2'].append(0)
        score['Event'].append('No Detection')
        score['Rally_Break'].append(0)
    flag = 0
    lastFrame=-1
    #input
    Check = df_track_ball[df_track_ball['ball_hit']==1]
    if(Check['delta_y'].tolist()[0]<0):
        last_player = 2
        other_player = 1
    else:
        last_player = 1
        other_player = 2
    scorelast = score[str(last_player)][-1]
    scoreother = score[str(other_player)][-1]
    Event = score['Event'][-1]
    last_coordinate = None
    last_bounce=None
    for i in range(frame_wanted, frame_nums):  
        if(df_track_ball['ball_hit'][i]==1):
                if(df_track_ball['delta_y'][i]>0):
                        last_player = 2
                        other_player=1
                if(df_track_ball['delta_y'][i]<0):
                        last_player = 1
                        other_player=2
        if i in bounces:
                    last_bounce = ball_track[i]
                    lastFrame = i
        
        if homography_matrices[i] is not None and math.isnan(ball_track[i][0]) and flag==0:
                x_min = max(l[i][0][0][0][0], l[i][1][0][0][0])
                x_max = min(l[i][2][0][0][0], l[i][3][0][0][0])
                y_min = max(l[i][0][0][0][1], l[i][2][0][0][1])
                y_max = min(l[i][1][0][0][1], l[i][3][0][0][1])
                
                
                    
                ball_point = last_bounce
                print(ball_point)
                ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                ball_point = cv2.perspectiveTransform(ball_point, homography_matrices[lastFrame])
                
                middle_court = (l[i][2][0][0][1]+l[i][3][0][0][1])/2
                up_threshold = (l[i][2][0][0][1]+middle_court)/2
                down_threshold = (l[i][3][0][0][1]+middle_court)/2
                
                if(ball_point[0][0][0]<=x_max and ball_point[0][0][0]>=x_min and ball_point[0][0][1]<=y_max and ball_point[0][0][1]>=y_min and ball_point[0][0][1]<up_threshold and ball_point[0][0][1]>down_threshold):
                    scorelast+=15 
                    Event = f'Player {other_player} Missed!'
                    
                else:
                    scoreother+=15
                    Event = f'Player {last_player} Foul!'
                score[str(last_player)].append(scorelast)
                score[str(other_player)].append(scoreother)
                flag = 1-flag
                score['Rally_Break'].append(1)
        elif homography_matrices[i] is None and flag==0:
                x_min = max(l[i-1][0][0][0][0], l[i-1][1][0][0][0])
                x_max = min(l[i-1][2][0][0][0], l[i-1][3][0][0][0])
                y_min = max(l[i-1][0][0][0][1], l[i-1][2][0][0][1])
                y_max = min(l[i-1][1][0][0][1], l[i-1][3][0][0][1])
                
                
                    
                ball_point = last_bounce
                print(ball_point)
                ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                ball_point = cv2.perspectiveTransform(ball_point, homography_matrices[lastFrame])
                # scorelast = score[str(last_player)][-1]
                # scoreother = score[str(other_player)][-1]
                middle_court = (l[i][2][0][0][1]+l[i][3][0][0][1])/2
                up_threshold = (l[i][2][0][0][1]+middle_court)/2
                down_threshold = (l[i][3][0][0][1]+middle_court)/2
                
                if(ball_point[0][0][0]<=x_max and ball_point[0][0][0]>=x_min and ball_point[0][0][1]<=y_max and ball_point[0][0][1]>=y_min and ball_point[0][0][1]<up_threshold and ball_point[0][0][1]>down_threshold):
                    scorelast+=15 
                    Event = f'Player {other_player} Missed!'
                else:
                    scoreother+=15
                    Event = f'Player {last_player} Foul!'
                score[str(last_player)].append(scorelast)
                score[str(other_player)].append(scoreother)
                flag = 1-flag
                score['Rally_Break'].append(1)
        elif homography_matrices[i] is not None and not math.isnan(ball_track[i][0]) and flag==0:
            score[str(last_player)].append(scorelast)
            score[str(other_player)].append(scoreother)
            flag = 0
            score['Rally_Break'].append(0)
        elif homography_matrices[i] is not None and not math.isnan(ball_track[i][0]) and flag==1:
            middle_court = (l[i][2][0][0][1]+l[i][3][0][0][1])/2
            if(ball_track[i][1]>middle_court):
                last_player = 2
                other_player = 1
            else:
                last_player = 1
                other_player = 2
            score[str(last_player)].append(scorelast)
            score[str(other_player)].append(scoreother)
            flag = 0
            Event = 'Rally Started'
            score['Rally_Break'].append(0)
        elif homography_matrices[i] is None and flag==1:
            score[str(last_player)].append(scorelast)
            score[str(other_player)].append(scoreother)
            score['Rally_Break'].append(0)
        elif (homography_matrices[i] is None or math.isnan(ball_track[i][0])) and flag==0:
            curr_y = ball_track[i-1][1]
            middle_court = (l[i][2][0][0][1]+l[i][3][0][0][1])/2
            up_threshold = (l[i][2][0][0][1]+middle_court)/2
            down_threshold = (l[i][3][0][0][1]+middle_court)/2
            if (curr_y<=down_threshold and curr_y>=up_threshold):
                scoreother+=15
            score[str(last_player)].append(scorelast)
            score[str(other_player)].append(scoreother)
            Event = 'Net Hit!' 
            flag = 1
            score['Rally_Break'].append(0)
        score['Event'].append(Event)
    return score