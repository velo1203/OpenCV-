import cv2
import mediapipe as mp
import keyboard
import json


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
finger = ['index','middle','ring','pinky']



# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def findNearNum(exList, values): #근사값을 찾는 함수
    answer = [0 for _ in range(2)] # answer 리스트 0으로 초기화

    minValue = min(exList, key=lambda x:abs(x-values))
    minIndex = exList.index(minValue)
    answer[0] = minIndex
    answer[1] = minValue
    
    return answer

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    left_finger = None
    right_finger = None
    left_finger_position = None
    right_finger_position = None
    success, image = cap.read()
    image = cv2.flip(image,0)
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
          key = keyboard.read_key()
          handedness = results.multi_handedness[hand_no].classification[0].label
          json_path = './info.json'
          json_data = {}
          with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
            
          if not key in json_data:
              break
        
          for i in range(1):
            #   print(f'{mp_hands.HandLandmark(i).name}:')

              if handedness == 'Left':
                left_index = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).INDEX_FINGER_TIP].x * width)
                left_middle = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).MIDDLE_FINGER_TIP].x * width)
                left_ring = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).RING_FINGER_TIP].x * width)
                left_pinky = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).PINKY_TIP].x * width)
                result_finger_left_index = findNearNum([left_index,left_middle,left_ring,left_pinky],json_data[key]['x'])
                
                left_finger = finger[result_finger_left_index[0]]
                left_finger_position = result_finger_left_index[1]
              
              elif handedness == 'Right':
                right_index = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).INDEX_FINGER_TIP].x * width)
                right_middle = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).MIDDLE_FINGER_TIP].x * width)
                right_ring = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).RING_FINGER_TIP].x * width)
                right_pinky = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).PINKY_TIP].x * width)
                result_finger_right_index = findNearNum([right_index,right_middle,right_ring,right_pinky],json_data[key]['x'])
                
                right_finger = finger[result_finger_right_index[0]]
                right_finger_position = result_finger_right_index[1]

      

          mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      key_hand = json_data[key]
      if not key_hand == None:
        print('----------------------')
        if left_finger_position == None:
          result_hand = 'right'
        elif right_finger_position == None:
          result_hand = 'left'
        else:
          result_hand = ['left','right'][findNearNum([left_finger_position,right_finger_position],json_data[key]['x'])[0]] 

        if result_hand == 'right':
          
          print(f'오른손 손가락 : {right_finger}, 좌표(x) : {right_finger_position}')
          result_finger = right_finger
        else:
          print(f'왼손 손가락 : {left_finger}, 좌표(x) : {left_finger_position}')
          result_finger = left_finger
        
      
        if key_hand['hand'] == result_hand and key_hand['finger'] == result_finger:
          print(f'{key} 키를 {result_hand}손, {result_finger}손가락으로 올바르게 치셨습니다')
        else:
          print(f'{key}키는 {result_hand}손, {result_finger}손가락이 아닌,\n ' + key_hand['hand'] + '손, ' + key_hand['finger'] + '손가락으로 치셨어야 합니다')
      
    image = cv2.flip(image,1)
    cv2.imshow('MediaPipe Hands',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()