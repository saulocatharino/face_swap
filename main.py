#!/usr/bin/python
# -*- coding: utf-8 -*-
# Desenvolvido por Saulo Catharino | saulocatharino@gmail.com

from tkinter import *
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from PIL import Image
from PIL import ImageTk
import cv2, time, threading, dlib, os
from threading import Thread
from saulo import face_swap_cropedimage
import numpy as np

DATA = [None, None]


def is_in_boundbox(x, y, boundbox):
    if (x < boundbox[0] or x > boundbox[0] + boundbox[2]):
        return False
    if (y < boundbox[1] or y > boundbox[1] + boundbox[3]):
        return False
    return True


def cvloop(run_event):
    global panelA, DATA

    model = "shape_predictor_68_face_landmarks.dat"
    tkMessageBox.showinfo("Instruções",
                           "1 - Carregue uma imagem válida \n2 - Clique no rosto do retângulo para trocar de face\n"
                           "3 - Clique fora do retângulo para destrocar")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)

    video_capture = cv2.VideoCapture(0)  # Caputra da Webcam

    panelA = Label()
    panelA.pack(side="left", padx=10, pady=10)

    while run_event.is_set():  # while the thread is active we loop
        ret, image = video_capture.read()

        img_ref = DATA[0]
        face_ref_rect = DATA[1]

        if face_ref_rect != None:
            swapped = face_swap_cropedimage(img_ref, face_ref_rect, image, detector, predictor)
            if swapped is not None:
                image = swapped

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()


def select_image():
    global panelB

    path = tkFileDialog.askopenfilename()

    if len(path) > 0:

        image = cv2.imread(path)

        if image is None:
            tkMessageBox.showerror("Erro", "Imagem inválida.")
        assert image is not None, "Not a valid image file"

        DATA[0] = np.copy(image);
        DATA[1] = None

        detector = dlib.get_frontal_face_detector()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        bound_boxes = []
        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            bound_boxes.append([x, y, w, h])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = ImageTk.PhotoImage(image)

        if panelB is None:

            panelB = Label(image=image)
            panelB.image = image
            panelB.pack(side="right", padx=10, pady=10)

        else:

            panelB.configure(image=image)
            panelB.image = image

        def printcoord(event, bound_boxes, rect_boxes):
            global DATA
            any_face = False
            for i, bound in enumerate(bound_boxes):
                if is_in_boundbox(event.x, event.y, bound):
                    DATA[1] = rect_boxes[i]
                    any_face = True
            if not any_face:
                DATA[1] = None

        panelB.bind("<Button 1>", lambda event: printcoord(event, bound_boxes, faces))


root = Tk()
root.title("Troca de Rostos")
this_dir = os.path.dirname(os.path.realpath(__file__))
imgicon = PhotoImage(file=os.path.join(this_dir, 'icon.png'))
root.tk.call('wm', 'iconphoto', root._w, imgicon)

panelA = None
panelB = None


btn = Button(root, text="Selecione a imagem", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()


def terminate():
    global root, run_event, action
    print("Encerrando opencv...")
    run_event.clear()
    time.sleep(1)
    root.destroy()


root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop()  # creates loop of GUI
