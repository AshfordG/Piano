import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import pygame, sys

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils
pygame.mixer.init()


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GREY = (93, 93, 93)
RED = (200, 0, 0)


class PianoKey:
    def __init__(self, note, color, note_file):
        self.color = color
        self.note = note
        self.note_file = note_file
        self.can_be_played = True
        self.bounds = []

    def draw(self, screen, coordinates):
        # cv2.rectangle(image, (pt1), (pt2), self.color, -1)
        pygame.draw.rect(screen, WHITE, (coordinates), width=0)
        

    def play_note(self):
        if self.can_be_played:
            print(self.note)
            pygame.mixer.Sound.play(self.note_file)


class Game:
    def __init__(self):
        self.keys = []
        self.landmarks = []
        self.x_shift = 0

        self.notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']
        
        #Initialize Keys
        self.Key_C = PianoKey('C', WHITE, pygame.mixer.Sound("sounds/C.wav"))
        self.Key_D = PianoKey('D', WHITE, pygame.mixer.Sound("sounds/D.mp3"))
        self.Key_E = PianoKey('E', WHITE, pygame.mixer.Sound("sounds/E.mp3"))
        self.Key_F = PianoKey('F', WHITE, pygame.mixer.Sound("sounds/F.mp3"))
        self.Key_G = PianoKey('G', WHITE, pygame.mixer.Sound("sounds/G.mp3"))
        self.Key_A = PianoKey('A', WHITE, pygame.mixer.Sound("sounds/A.wav"))
        self.Key_B = PianoKey('B', WHITE, pygame.mixer.Sound("sounds/B.wav"))
        self.Key_C2 = PianoKey('C2', WHITE, pygame.mixer.Sound("sounds/C.mp3"))   

        self.keys.append(self.Key_C)
        self.keys.append(self.Key_D)
        self.keys.append(self.Key_E)
        self.keys.append(self.Key_F)
        self.keys.append(self.Key_G)
        self.keys.append(self.Key_A)
        self.keys.append(self.Key_B)
        self.keys.append(self.Key_C2)

        #Hand landmarks
        self.index = HandLandmarkPoints.INDEX_FINGER_TIP.value
        self.middle = HandLandmarkPoints.MIDDLE_FINGER_TIP.value
        self.ring = HandLandmarkPoints.RING_FINGER_TIP.value
        self.pinky = HandLandmarkPoints.PINKY_TIP.value
        self.thumb = HandLandmarkPoints.THUMB_TIP.value

        self.landmarks.append(self.index)
        self.landmarks.append(self.middle)
        self.landmarks.append(self.ring)
        self.landmarks.append(self.pinky)
        self.landmarks.append(self.thumb)

        # Create Hand Detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        self.video = cv2.VideoCapture(1)

        while not self.video.isOpened():
            continue

        image = self.video.read()[1]
        print(image.shape)
        self.screen = pygame.display.set_mode((800, 500))
        pygame.display.set_caption("Hello World")

    
    def draw_landmarks_on_hand(self, image, detection_result):
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())


    
    def check_key_press(self, finger_x, finger_y, key):
        if finger_x > key.bounds[0] and finger_x < key.bounds[1] and finger_y > 300 and finger_y < 400:
            key.play_note()
            key.can_be_played = False
        else:
            key.can_be_played = True


    # def track_finger(self, image, detection_result):
    #     imageHeight, imageWidth = image.shape[:2]
    #     hand_landmarks_list = detection_result.hand_landmarks
    #     for idx in range(len(hand_landmarks_list)):
    #         hand_landmarks = hand_landmarks_list[idx]
            
    #         # Get the coordinates of just the index finger 
    #         index_finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
    #         middle_finger = hand_landmarks[HandLandmarkPoints.MIDDLE_FINGER_TIP.value]
    #         # Map the coordinates back to screen dimensions 
    #         pixelCoordinates = DrawingUtil._normalized_to_pixel_coordinates(index_finger.x, index_finger.y, imageWidth, imageHeight)
    #         middleCoordinates = DrawingUtil._normalized_to_pixel_coordinates(middle_finger.x, middle_finger.y, imageWidth, imageHeight)
    #         if pixelCoordinates:
    #             pygame.draw.circle(self.screen, BLUE, (pixelCoordinates[0], pixelCoordinates[1]), 10, 2)
    #             pygame.draw.circle(self.screen, BLUE, (middleCoordinates[0], middleCoordinates[1]), 10, 2)
    #             return pixelCoordinates
        
    def track_finger(self, image, detection_result, landmark):
        imageHeight, imageWidth = image.shape[:2]
        hand_landmarks_list = detection_result.hand_landmarks
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Get the coordinates of just the index finger 
            
            finger = hand_landmarks[landmark]
            # Map the coordinates back to screen dimensions 
            pixelCoordinates = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)
            if pixelCoordinates and landmark == self.thumb:
                pygame.draw.circle(self.screen, BLUE, (pixelCoordinates[0], pixelCoordinates[1] - 175), 10, 2)
                return pixelCoordinates 
            elif pixelCoordinates and landmark == self.pinky:
                pygame.draw.circle(self.screen, BLUE, (pixelCoordinates[0], pixelCoordinates[1] - 100), 10, 2)
                return pixelCoordinates
            elif pixelCoordinates:
                pygame.draw.circle(self.screen, BLUE, (pixelCoordinates[0], pixelCoordinates[1] ), 10, 2)
                return pixelCoordinates

            

    def run(self):
        while self.video.isOpened():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            #Clear Screen
            self.screen.fill(GREY)

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes mirrored - flip it
            image = cv2.flip(image, 1)
            
            #Draw Piano
            x = 0
            for key in self.keys:
                key.bounds.clear()
                key.draw(self.screen, (100 + x, 100, 75, 300))
                key.bounds.append(100 + x)
                key.bounds.append(175 + x)
                x += 85
            pygame.draw.rect(self.screen, BLACK, (100, 50, 670, 50), width=0)
            pygame.draw.line(self.screen, RED, (100, 300), (770, 300), 4)
            

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)
            for finger in self.landmarks:
                coordinates = self.track_finger(image, results, finger)
                if coordinates != None:
                    for key in self.keys:
                        self.check_key_press(coordinates[0], coordinates[1], key)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Hand Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            pygame.display.flip()
        
        # Release our video and close all windows
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()