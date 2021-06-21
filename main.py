# Importing some python packages
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

STORE_TIME = True # if you want to store time of the person
JARVIS_SPEAK = True # if you want a machine to speak about the person arrival

path = 'Training_images' # path where we will keep our training images of our relatives and family person
images = [] # a container to store all images
classNames = [] # a container to store all image name
myList = os.listdir(path) # this command brings out name of all the files from specified directory

# Now, we are reading all the image and storing its name one by one and Python 'for loop' is helping us
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(f"We have got {','.join(classNames)} in our trained image folder")


def findEncodings(images):
    '''

    :param images: a list of images
    :return: a list containing facial encoding of given images
    '''
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def gtts_play(mytext):

    language = 'en'
    myobj = gTTS(text=mytext, lang=language, tld="co.in", slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("machine_voice.mp3")
    play(AudioSegment.from_mp3('machine_voice.mp3'))


def jarvis_talk(name_list):
    '''

    :param name_list: list of name
    :return: Machine speaks with instructed command
    '''
    if len(name_list)==1:
        print(f"{name_list[0]} is at the door")
        gtts_play(f"{name_list[0]} is at the door")

    if len(name_list)==2:
        print(f"{name_list[0]} and {name_list[1]} is at the door")
        gtts_play(f"{name_list[0]} and {name_list[1]} is at the door")

    if len(name_list)>2:
        print(f"{name_list[0]} and {len(name_list[1:])} other person is at the door")
        gtts_play(f"{name_list[0]} and {len(name_list[1:])} other person is at the door")


def markAttendance(name):
    '''

    :param name: a person name
    :return: The function stores the person name in a CSV
    '''
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                dtstring = datetime.now().strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtstring}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr


# Our Facial Recognition gets trained on the given images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0) # to open a webcam in python
print(cv2.CAP_PROP_FPS) # this will tell you the frame per second of your webcam
count = 0

while True: # a while loop to run the webcam until interruption
    success, img = cap.read() # take an image frame from a webcam
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # to resize the image
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # to recolor the image from BGR to RGB format (R:red, G:green, B:blue)

    facesCurFrame = face_recognition.face_locations(imgS) # to locate the face in the image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) # to extract the facial encoding (facial points)

    face_list = []
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # a loop to match image with the trained images
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            face_list.append(name)

        if count % 50 == 0 and STORE_TIME:
            for person_name in face_list:
                markAttendance(person_name)

        if count % 70 == 0 and JARVIS_SPEAK:
            jarvis_talk(face_list)

    count = count + 1

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)