import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import smtplib
from email.message import EmailMessage
import winsound
import geocoder
#email notification
def email_alert(subject,body,to):
    frequency = 2500
    duration = 1000  
   
    winsound.Beep(frequency, duration)
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to']= to
    
    user = "sivaramakrishna052@gmail.com"
    msg['from']=user
    password= "rjfjiocybckhuvlo"
    
    server = smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(user,password)
    server.send_message(msg)
    server.quit()
    


    
# creating the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the training data 
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('Downloads/train/fire_dataset/', target_size=(64, 64), batch_size=32, class_mode='binary')

# Train the model
model.fit( train_generator,
    steps_per_epoch=train_generator.n//train_generator.batch_size,
    epochs=15)


def predict_fire_from_camera(threshold=0.5):
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)  # Resize the frame
        frame = frame / 255.0 
        frame = np.expand_dims(frame, axis=0)
        prediction = model.predict(frame)  # Make a prediction using the trained model
      
        
        
        #layout of camera on screen
        cv2.imshow('Camera Feed', frame[0])  
        if prediction[0][0] > threshold:
            print('Fire Detected')
            #geo location
            g = geocoder.ip('me')
            current_location = g.latlng
            message = f"Fire has been detected at the following location: {current_location}"
            email_alert("FIRE ALER",message,"lukboss979@gmail.com")
            
            
        else:
            print('No Fire Detected')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # to close the camera
    cv2.destroyAllWindows() 


predict_fire_from_camera(threshold=0.3)
