import cv2
import smtplib
import time
from time import sleep
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
flagi = 1
# Email Variables
SMTP_SERVER = 'smtp.gmail.com'  # Email Server (don't change!)
SMTP_PORT = 587  # Server Port (don't change!)
GMAIL_USERNAME = 'homealonerobot@gmail.com'  # change this to match your gmail account
GMAIL_PASSWORD = 'Homealonerobotteam14'  # change this to match your gmail password


class Emailer:
    def sendmail(self, recipient, subject, content, image):
        # Create Headers
        emailData = MIMEMultipart()
        emailData['Subject'] = subject
        emailData['To'] = recipient
        emailData['From'] = GMAIL_USERNAME
        #Attach our text data
        emailData.attach(MIMEText(content))
        #Create our Image Data from the defined image
        imageData = MIMEImage(open(image, 'rb').read(), 'jpg')
        imageData.add_header('Content-Disposition', 'attachment; filename="image.jpg"')
        emailData.attach(imageData)

        # Connect to Gmail Server
        session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        session.ehlo()
        session.starttls()
        session.ehlo()

        # Login to Gmail
        session.login(GMAIL_USERNAME, GMAIL_PASSWORD)

        # Send Email & Exit
        session.sendmail(GMAIL_USERNAME, recipient, emailData.as_string())
        session.quit


sender = Emailer()

sendTo = 'ahmedgs2009@gmail.com'
emailSubject = "Alert"
emailContent = "There is an Intruder in the house"
# Sends an email to the "sendTo" address with the specified "emailSubject" as the subject and "emailContent" as the email content.
#sender.sendmail(sendTo, emailSubject, emailContent)
sendimage = cv2.imread('test.jpg',1)
sleep(10)
#cv2.imshow('test', sendimage)
#sender.sendmail(sendTo, emailSubject, emailContent, sendimage)
# Face Detection Code
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_lowerbody.xml')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_vrepcascade.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    #cv2.imwrite('Intruder.jpg', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.01, 1, 0, [100, 100])
    # faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Make square around face and eyes for capturing intruder face
    #if flagi == 1:
    #cv2.imwrite('Intruder.jpg', frame)  # save frame as JPEG file
        #flagi = 0
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 'Intruder', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        x = 1

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('img', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

capture.release()
cv2.destroyAllWindows()
