import os  # accessing the os functions
import check_camera
import Capture_Image
import Train_Image

# creating the title bar function

def title_bar():
    os.system('cls')  # for windows

    # title of the program

    print("\t**********************************************")
    print("\t***** Face Recognition Attendance System *****")
    print("\t**********************************************")


# creating the user main menu function

def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Check Camera")
    print("[2] Capture Faces")
    print("[3] Load Model and Mark Attendance")
    print("[4] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                checkCamera()
                break
            elif choice == 2:
                CaptureFaces()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-5")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-5\n Try Again")
    exit

def checkCamera():
    check_camera.camer()
    key = input("Enter any key to return main menu")
    mainMenu()

def CaptureFaces():
    Capture_Image.takeImages()
    key = input("Enter any key to return main menu")
    mainMenu()

def Trainimages():
    Train_Image.TrainImages()
    key = input("Enter any key to return main menu")
    mainMenu()

# ---------------main driver ------------------
mainMenu()
