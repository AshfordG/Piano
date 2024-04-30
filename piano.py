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

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)




class PianoKey:
    def __init__(self, note, color, screen_width=600, screen_height=400):
        self.color = color
        self.note = note
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def draw(self, image, pt1, pt2):
        cv2.rectangle(image, (pt1), (pt2), self.color, -1)

    def play_note():
        pass


class Game:
    def __init__(self):
        self.keys = []
        self.x_shift = 0

        self.notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']
        #Initialize Keys
        self.Key_C = PianoKey('C', WHITE)
        self.Key_D = PianoKey('D', WHITE)
        self.Key_E = PianoKey('E', WHITE)
        self.Key_F = PianoKey('F', WHITE)
        self.Key_G = PianoKey('G', WHITE)
        self.Key_A = PianoKey('A', WHITE)
        self.Key_B = PianoKey('B', WHITE)
        self.Key_C2 = PianoKey('C2', WHITE)   

        self.keys.append(self.Key_C)
        self.keys.append(self.Key_D)
        self.keys.append(self.Key_E)
        self.keys.append(self.Key_F)
        self.keys.append(self.Key_G)
        self.keys.append(self.Key_A)
        self.keys.append(self.Key_B)
        self.keys.append(self.Key_C2)

        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
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

    def check_key_press(self, finger_x, finger_y, key, image):
        #A key will count as 'played' if the x coordinate is within the keys bounds and y coordinate is below ____
        if finger_x > key.x - 10 and finger_x < key.x + 10 and finger_y > key.y - 10 and finger_y < key.y + 10:
            key.play_note()

    def track_finger(self, image, detection_result):
        imageHeight, imageWidth = image.shape[:2]
        hand_landmarks_list = detection_result.hand_landmarks
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Get the coordinates of just the index finger 
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            # Map the coordinates back to screen dimensions 
            pixelCoordinates = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)
            if pixelCoordinates:
                return pixelCoordinates
                # Draw the circle around index finger
                # pygame.draw.circle(self.screen, BLUE, (pixelCoordinates[0], pixelCoordinates[1]), 25, 2)
                # cv2.circle(image, (pixelCoordinates[0], pixelCoordinates[1]), 25, BLUE, 5)

                # self.check_key_press(pixelCoordinates[0], pixelCoordinates[1], self.green_enemy, image)




    def run(self):
        while self.video.isOpened():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            #Clear Screen
            self.screen.fill((0,0,0))

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes mirrored - flip it
            image = cv2.flip(image, 1)

            
            #Draw Piano
            # pygame.rectangle(self.screen, (600, 350), (1200, 400), BLACK, -1)
            x = 0
            for key in self.keys:
                # key.draw(self.screen, (600 + x, 400), (675 + x, 700))
                cv2.rectangle(image, (600 + x, 400), (675 + x, 700), BLACK, 4)
                x += 75
            

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)
            coordinates = self.track_finger(image, results)
            if coordinates != None:
                pygame.draw.circle(self.screen, BLUE, (coordinates[0], coordinates[1]), 10, 2)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Tracking', image)
            

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            # pygame.draw.circle(self.screen, BLUE, (100, 100), 15, 2)
            pygame.display.flip()
        
        # Release our video and close all windows
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()