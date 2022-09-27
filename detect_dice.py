import cv2
import numpy as np
from sklearn import cluster
from matplotlib import pyplot as plt

"""
Maya Krolik
September 2022
AH Computer Science
"""

# ------------------------------------------------------------------------------
def get_dots(frame):
    """ take a small shortcut by using opencv's built in blob detection :) 
        blur each frame and detect any noticable spots """

    parameters = cv2.SimpleBlobDetector_Params()

    detector = cv2.SimpleBlobDetector_create(parameters)
    
    # blur image
    frame_blurred = cv2.medianBlur(frame, 7)
    # convert image from BGR to grayscale
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    # use pre-made dot detection to detect dots
    dots = detector.detect(frame_gray)

    return dots

# ------------------------------------------------------------------------------
def count_dice_from_dots(dots):
    """ Get center of all clustered dots and # of dots on dice """

    processed_dots = []
    for i in dots:
        # find center of area of interest
        position = i.pt
        if position != None:
            processed_dots.append(position)

    processed_dots = np.asarray(processed_dots) # convert list to np array

    if len(processed_dots) > 0: # if dots were detected, group them together
        
        # sed maximum distance apart that groups can be in order to be clumped
        # together, as well as the minimum number of groups
        grouping = cluster.DBSCAN(eps=60, min_samples=1).fit(processed_dots)

        # grouping.labels will return the maximum label of the groups given
        # counting starts at 0, so adding 1 will give you the total number of
        # detected dice
        number_of_dice = max(grouping.labels_) + 1
        dice = []

        # Calculate center of each dice, the average between all a dice's dots
        for i in range(number_of_dice):
            dots_on_dice = processed_dots[grouping.labels_ == i]
            center_of_dice = np.mean(dots_on_dice, axis=0)
            # add number of dots on dice, center of dice to list
            dice.append([len(dots_on_dice), *center_of_dice])
        return dice
    else:
        return []

# ------------------------------------------------------------------------------
def draw_information(frame, dice, dots):
    """ create overlay text depending on the # of dots and position """

    # Overlay blobs
    for i in dots:
        position = i.pt
        radius = i.size / 2 # divide diameter in half to get radius
        # draw circle around each dot 
        cv2.circle(frame, (int(position[0]), int(position[1])),
                   int(radius), (255, 0, 0), 2)

    # Overlay dice number
    for j in dice:
        # Get textsize, returns size of box that can mazimize font
        textsize = cv2.getTextSize(
            str(j[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
        # put text in center of dice with label of dice number
        cv2.putText(frame, str(j[0]),
                    (int(j[1] - textsize[0] / 2),
                     int(j[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

# ------------------------------------------------------------------------------
def record_values(dice):
    """ when called, record currently displayed values,
        allows for multiple die """

    events = []
    for i in dice:
       events.append(str(i[0])) 
    print("recorded values!")
    return events

# ------------------------------------------------------------------------------
def update_frequencies(dice_number, events):
    """ updates lists sorting types of dice numbers recorded thus far and
        their frequencies """

    for event in events:
        dice_number.append(event)

# ------------------------------------------------------------------------------
def make_plot(dice_numbers):
    """ create a plot of the frequencies of each die roll """

    dice_numbers.sort() # order data numerically

    # create plot with appropreate labeles
    plt.figure()
    plt.title("Dice Histogram")
    plt.xlabel("Dice number")
    plt.ylabel("Frequency")

    # plot frequency of each number
    # setting density to false will make y axis count frequency
    # setting bins to 6 will make 6 even partitions on x axis (1-6)
    plt.hist(dice_numbers, density=False, bins=6)
    plt.show()

# ------------------------------------------------------------------------------
def set_camera():
    """ adjust code to computer depending if extra camera is plugged in """

    user_response = input("Are you using an external usb camera? (Y/n) ")
    if user_response == "Y" or user_response == "y" or user_response == "yes" or user_response == "Yes":
        return "1"
    if user_response == "n" or user_response == "N" or user_response == "no" or user_response == "No":
        return "0"
    else:
        print("Please enter a valid input")
        return set_camera()

# ------------------------------------------------------------------------------
def main():
    dice_numbers = []
    
    # Initialize a video feed, 0 if it is the only/primary/built in camera, 1 if
    # it is a plugged in webcam
    camera_used = int(set_camera())
    image = cv2.VideoCapture(camera_used)

    # if camera doesnt work or if image cannot be read, try other camera
    if not image.isOpened() or image == None:
        if camera_used == 0:
            # should never run because if you dont have a built-in camera, you
            # probably dont have a plugged in one that works under index of 1
            print("\nYour built-in camera was not detected, switched to external")
            image = cv2.VideoCapture(1)
        else: # external camera failed
            print("\nYour external camera was not detected, switched to built-in")
            image = cv2.VideoCapture(0)
        # if still doesnt work
        if not image.isOpened():
            raise IOError("no camera detected")

    while(True): # while camera is running

        # Grab the latest image from the video feed
        _, frame = image.read()
        dots = get_dots(frame)
        
        # if any dots were detected, proceed with counting dice and displaying info
        if len(dots) > 0:
            dice = count_dice_from_dots(dots)
            draw_information(frame, dice, dots)

        # display image
        cv2.imshow("Counting Dice!", frame)

        res = cv2.waitKey(1)

        # record value if "r" is pressed
        if res & 0xFF == ord('r'):
            values = record_values(dice)
            update_frequencies(dice_numbers, values)
            
        # Stop if "q" is pressed
        if res & 0xFF == ord('q'):
            break

    # When finished, release the image
    image.release()
    cv2.destroyAllWindows()
    # from the collected data, make a histogram of the occurance of each roll
    make_plot(dice_numbers)

main()