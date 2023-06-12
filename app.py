"""

    La seguente Demo deve 

        1. Caricare FaceNet512 con i relativi pesi

        2. Per ogni user presente nel nostro database (user = folder)

            2.1 Caricare ogni singola immagine
            2.2 Tramite MTCNN (presente in DeepFace) estrarre il volto | Assunzione -> 1 volto per immagine
            2.3 Tramite FaceNet512 ottenere l'embedding del volto e salvarlo in una list dedicata all'utente
            2.4 Calcolare la media degli embedding per ogni utente e salvarla in un file .npy

        3. Caricare il file .npy per ogni utente

        4. Per ogni frame del video

            4.1 Tramite MTCNN (presente in DeepFace) estrarre i volti 
            4.2 Tramite FaceNet512 ottenere l'embedding dei volti
            4.3 Per ogni embedding calcolare la distanza euclidea con la media degli embedding degli utenti
            4.4 Se la distanza minore è inferiore ad una certa soglia, allora l'utente è presente nel video
            4.5 Mostrare a video la presenza dell'utente


        How to run:

            1. git clone https://github.com/serengil/deepface.git
            2. create a folder with your name in the database folder and put your images or videos inside it (the image need to contain only your face)
            2. python3 app.py

        
"""

from model.Facenet512 import loadModel
from deepface import DeepFace
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
import os
from rich.console import Console

console = Console()
backends = "mtcnn"
database = "database/"
weights = "weights/facenet512_weights.h5"
target_size = (160, 160)

console.rule("[bold red]Inizializing [/bold red]")

# Cartelle utili al funzionamento

os.makedirs("log", exist_ok=True)
os.makedirs("database/embedded", exist_ok=True)

# clear the missing_faces.log file
with open("log/missing_faces.log", "w") as f:
    f.write("")
    

console.log("Cartelle utili al funzionamento create")
# Cartelle utili al funzionamento



# 1. Caricare FaceNet512 con i relativi pesi

model = loadModel(weights_path=weights)
console.log("FaceNet512 loaded")

# 2. Per ogni user presente nel nostro database (user = folder)

console.rule("[bold red] Generazione embedded [/bold red]")

for user in os.listdir(database):

    if user == "embedded":
        continue

    if os.path.exists(database +  "embedded" + "/" + user + ".npy"):
        console.log("l'utente [bold red]" + user + "[/bold red] è già stato analizzato")
        continue
    

    # get the number of images for each user

    num_images = len(os.listdir(database + "/" + user))
    checked_images = 0

    for image in os.listdir(database + "/" + user):

        img_path = database + "/" + user + "/" + image
        user_embeddeds = list()


        # check if is video

        if image.endswith(".MP4") or image.endswith(".avi") or image.endswith(".mov"):

            console.log("[bold red]" + img_path + "[/bold red] è un video")

            # get the number of frames

            cap = cv2.VideoCapture(img_path)

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(num_frames):

                # get the frame

                cap.set(cv2.CAP_PROP_POS_FRAMES, i)

                ret, frame = cap.read()

                if frame is None:
                    continue
                
                # extract the face
                try:
                    face_objs = DeepFace.extract_faces(frame, detector_backend = backends, target_size = target_size, verbose=False)
                except Exception as e:
                    with open("log/missing_faces.log", "a") as f:
                        f.write("il " + str(i) + " frames " + " del video " + image + " non contiene volti" + "\n")
                        continue
                
                user_face = face_objs[0]['face']

                user_face_expanded = np.expand_dims(user_face, axis=0)

                prediction = model.predict(user_face_expanded, verbose=0)[0].tolist()
                user_embeddeds.append(prediction)

                if i % 10 == 0 or i == num_frames - 1 or i  == 1: 
                    console.log("[bold red]" + str(i) + " / " + str(num_frames) +  "[/bold red]" + " frames per il video [bold red]" + image + "[/bold red]")

            cap.release()

        else:
            try:
                face_objs = DeepFace.extract_faces(img_path, detector_backend = backends, target_size = target_size, verbose=False)
            except Exception as e:
                with open("log/missing_faces.log", "a") as f:
                    f.write(img_path + "\n")
                    checked_images += 1
                    continue

        checked_images += 1

        if checked_images % 10 == 0 or checked_images == num_images or checked_images == 1: 
            console.log("[bold red]" + str(checked_images) + " / " + str(num_images) +  "[/bold red]" + " per l'utente [bold red]" + user + "[/bold red]")

        # if the image not contains a face, then skip it

        user_face = face_objs[0]['face']    #  embedded 
        user_face_expanded = np.expand_dims(user_face, axis=0)

        # 2.3 Tramite FaceNet512 ottenere l'embedding del volto e salvarlo in una list dedicata all'utente
        prediction = model.predict(user_face_expanded, verbose=0)[0].tolist()
        user_embeddeds.append(prediction)
        

    # 2.4 Calcolare la media degli embedding per ogni utente e salvarla in un file .npy

    user_embedded = np.mean(user_embeddeds, axis=0)

    np.save(database + "/" +  "embedded" + "/" + user + ".npy", user_embedded)

    console.log("l'utente [bold red]" + user + "[/bold red] è stato analizzato")

# 3. Caricare il file .npy per ogni utente

users_embedded = dict()

for user in os.listdir(database):

    if user == "embedded":
        continue

    user_embedded = np.load(database + "/" +  "embedded" + "/" + user + ".npy")

    assert user_embedded.shape == (512,) , "The shape of the user embedding is not correct"

    users_embedded[user] = user_embedded


console.rule("[bold red] Setup the nearest neighbor search model [/bold red]")

embeddings = [users_embedded[user] for user in users_embedded]
users = list(users_embedded.keys())

nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn_model.fit(embeddings)

console.log("Nearest neighbor search model loaded")



# 4. Ottenere il video dalla webcam

console.rule("[bold red]Inizializing Live Analysis [/bold red]")

cap = cv2.VideoCapture(0) #webcam


frame_counter = 0

while True:

    frame_counter += 1

    ret, frame = cap.read()

    cv2.flip(frame, 1, frame)

    # visualizzare un countdown di 20 secondi prima di iniziare l'analisi

    cv2.putText(frame, str(5 - int(frame_counter/20)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    if frame_counter == 100:

        try :
            faces = DeepFace.extract_faces(frame, detector_backend = backends, target_size = target_size, verbose=False)
        except Exception as e:
            console.log("[bold red]Nessun Volto è stator rilevato[/bold red]")

            # Attendi che l'utente prema il tasto q per riprendere l'analisi
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    frame_counter = 0
                    break
            continue

        for face in faces:

            # evita di analizzare volti troppo piccoli
            if face['facial_area']['w'] < 100 or face['facial_area']['h'] < 100:
                continue
            
            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']

            face_expanded = np.expand_dims(face['face'], axis=0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 4.3 Per ogni embedding calcolare la distanza euclidea con la media degli embedding degli utenti
            prediction = model.predict(face_expanded, verbose=0)[0].tolist()

            min_distance = 1000
            min_user = None

            """
            for user in users_embedded:

                user_embedded = users_embedded[user]

                distance = np.linalg.norm(prediction - user_embedded)

                if distance < min_distance:
                    min_distance = distance
                    min_user = user
            """

            distance, index = nn_model.kneighbors([prediction])
            
            # 4.4 Se la distanza minore è inferiore ad una certa soglia, allora l'utente è presente nel video
            
            console.log(str(distance[0][0]) + " from " + str(users[index[0][0]]))
            
            if distance[0][0] < 25:
                
                min_user = users[index[0][0]]
                min_distance = distance[0][0]

                display_img = cv2.imread("database/%s/%s-10.jpg" % (min_user, min_user))
                pivot_img_size = 112
                display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                
                # if the user is on top of the screen, then put the name below the face
                if y < 112:
                    frame[y+h+25:y+h+25+pivot_img_size, x:x+pivot_img_size] = display_img
                else:
                    frame[y-112-25:y-25, x:x+pivot_img_size] = display_img


                # put the name of the user
                cv2.putText(frame, min_user, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(str(100 - round(min_distance, 2)))+ "%", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:

                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, str(str(100 - round(min_distance, 2)))+ "%", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            frame_counter = 0
        


    cv2.imshow("frame", frame)

    if frame_counter == 0:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.waitKey(1)
    







    





