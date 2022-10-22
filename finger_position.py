import cv2
import mediapipe as mp
import keyboard
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


with open('position.json', "r") as json_file:
  json_data = json.load(json_file)

with open('info.json','w') as outfile:
  json.dump(json_data,outfile,indent=4)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("카메라를 찾을 수 없습니다.")
      # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
      continue

    # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 이미지에 손 주석을 그립니다.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())



        #키보드 입력 json 처리
        key = keyboard.read_key()
        json_path = './info.json'
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
      
        finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
        finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

        data = {
          'x':finger_x,
          'y':finger_y
        }

        if key in json_data:
          json_data[key]['x'] = finger_x
          json_data[key]['y'] = finger_y
        else:
          json_data[key] = data

        with open(json_path,'w') as outfile:
          json.dump(json_data,outfile,indent=4)


    #보기 편하게 이미지를 좌우 반전합니다.
    image = cv2.flip(image,0)
    image = cv2.flip(image,1)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()