import cv2
import numpy as np
from sklearn import cluster
from matplotlib import pyplot as plt

# ------------------------------------------------------------------------------
def get_dots(frame):
    """ take a small shortcut by using opencv's built in blob detection :) 
        blur each frame and detect any noticable spots """
    params = cv2.SimpleBlobDetector_Params()

    detector = cv2.SimpleBlobDetector_create(params)
    
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
        grouping = cluster.DBSCAN(eps=40, min_samples=1).fit(processed_dots)

        # grouping.labels will return the maximum label of the groups given
        # counting starts at 0, so adding 1 will give you the total number of
        # detected dice
        number_of_dice = max(grouping.labels_) + 1
        dice = []

        # Calculate center of each dice, the average between all a dice's dots
        for i in range(number_of_dice):
            X_dice = processed_dots[grouping.labels_ == i]
            center_of_dice = np.mean(X_dice, axis=0)
            # add number of dots on dice, center of dice to list
            dice.append([len(X_dice), *center_of_dice])
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
def main():
    dice_numbers = []
    
    # Initialize a video feed, 0 if it is the only/primary/built in camera, 1 if
    # it is a plugged in webcam
    cap = cv2.VideoCapture(0)

    while(True): # while camera is running

        # Grab the latest image from the video feed
        ret, frame = cap.read()
        dots = get_dots(frame)
        
        # if any dots were detected, proceed with counting dice and displaying info
        if len(dots) > 0:
            dice = count_dice_from_dots(dots)
            out_frame = draw_information(frame, dice, dots)

        # display image
        cv2.imshow("frame", frame)

        res = cv2.waitKey(1)

        # record value if "r" is pressed
        if res & 0xFF == ord('r'):
            values = record_values(dice)
            update_frequencies(dice_numbers, values)
            
        # Stop if "q" is pressed
        if res & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # from the collected data, make a histogram of the occurance of each roll
    make_plot(dice_numbers)

main()