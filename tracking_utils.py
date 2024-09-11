import numpy as np
import math

def calculate_distance(pos1, pos2):
    x1 = pos1[0]
    x2 = pos2[0]
    y1 = pos1[1]
    y2 = pos2[1]
    if math.isnan(x1) or math.isnan(x2):
        return 0       
    else: 
        return ((x2-x1)**2+(y2-y1)**2)**0.5


def calculate_speed(position1, position2, fps):
    """
    Calculate the speed between two positions given the FPS of the video.
    :param position1: (x, y) position at time t1
    :param position2: (x, y) position at time t2
    :param fps: Frames per second of the video
    :return: Speed in pixels per second
    """
    if math.isnan(position1[0]) or math.isnan(position2[0]):
        return 0
    else:
        distance = calculate_distance(position1, position2)
        speed = distance / fps
        return speed

def angle_and_quad(pos1, pos2):
    quad = ""
    angle = 0
    if pos1[0] is None or pos2[0] is None:
        return angle, quad
    else:
        x1 = pos1[0]
        x2 = pos2[0]
        y1 = pos1[1]
        y2 = pos2[1]
        dx = x2-x1
        dy = y2-y1
        if dx == 0: dx = 1e-5 
        quad = ""
        angle = np.arctan(abs(dy/dx))
        converter = 180/np.pi
        
        if angle == 0 and dx>0: # x-axis
            quad = "East"
            return angle*converter, quad
        if angle == 0 and dx<0: #x-axis
            quad = "West"
            return angle*converter, quad
            
        if dx == 0 and dy>0: #y-axis
            quad = "South"
            return angle*converter, quad 
        
        if dx == 0 and dy<0: #y-axis
            quad = "North"
            return angle*converter, quad 
        
        if(dx>=0 and dy>=0): #First Quadrant
            if angle >=np.pi/4:
                quad = "Degree East of South"
            else:
                quad = "Degree South of East"
            return angle*converter, quad
        if(dx<=0 and dy>=0): #Second Quadrant
            if angle >=np.pi/4:
                quad = "Degree West of South"
            else:
                quad = "Degree South of West"
            return angle*converter, quad
        if(dx<=0 and dy<=0): # Third Quadrant
            if angle >=np.pi/4:
                quad = "Degree West of North"
            else:
                quad = "Degree North of West"
            return angle*converter, quad
        if(dx>=0 and dy<=0): #Fourth Quadrant+
            if angle >=np.pi/4:
                quad = "Degree East of North"
            else:
                quad = "Degree North of East"
            return angle*converter, quad        
        

# def calculate_direction(position1, position2):
#     """
#     Calculate the direction of movement between two positions and return as cardinal direction with angle.
#     :param position1: (x, y) position at time t1
#     :param position2: (x, y) position at time t2
#     :return: Direction as a string
#     """
#     if position1 is None or position2 is None:
#         return 0
#     delta = np.array(position2) - np.array(position1)
#     angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
#     angle = (angle + 360) % 360  # Normalize angle to [0, 360) degrees
    
#     # Determine the direction
#     directions = ["North", "East", "South", "West"]
#     direction_angle = 45  # 360 / 8 directions
    
#     index = int((angle + direction_angle / 2) % 360 // direction_angle) % 8
#     direction = directions[index]
    
#     # Format direction with angle
#     direction_str = f"{direction} {angle:.2f} degrees"
#     return direction_str


# def track_ball_speed_and_direction(ball_track, fps):
#     ball_speeds = []
#     ball_directions = []
    
#     for i in range(1, len(ball_track)):
#         if ball_track[i-1] is not None and ball_track[i] is not None:
#             speed = calculate_speed(ball_track[i-1], ball_track[i], fps)
#             direction = calculate_direction(ball_track[i-1], ball_track[i])
#         else:
#             speed = None
#             direction = None
        
#         ball_speeds.append(speed)
#         ball_directions.append(direction)
    
#     return ball_speeds, ball_directions

# def track_player_speed_and_direction(players, fps):
#     player_speeds = []
#     player_directions = []
#     for i in range(1, len(players)):
#         frame_speeds = []
#         frame_directions = []
#         for j in range(len(players[i])):
#             if players[i][j][1] and players[i-1][j][1]:
#                 speed = calculate_speed(players[i-1][j][1], players[i][j][1], fps)
#                 direction = calculate_direction(players[i-1][j][1], players[i][j][1])
#                 frame_speeds.append(speed)
#                 frame_directions.append(direction)
#             else:
#                 frame_speeds.append(None)
#                 frame_directions.append(None)
#         player_speeds.append(frame_speeds)
#         player_directions.append(frame_directions)
#     return player_speeds, player_directions
