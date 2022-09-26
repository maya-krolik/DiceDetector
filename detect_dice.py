import cv2
import numpy as np
from sklearn import cluster
from matplotlib import pyplot as plt


params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create(params)

# ------------------------------------------------------------------------------
def get_blobs(frame):
    """ blur each frame and detect any noticable spots """

    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs

# ------------------------------------------------------------------------------
def get_dice_from_blobs(blobs):
    """ Get center of all clusterd dots and # of dots on dice """

    X = []
    for i in blobs:
        # find center of area of interest
        position = i.pt

        if position != None:
            X.append(position)

    X = np.asarray(X) # convert list to np array

    if len(X) > 0:
        # Important to set min_sample to 0, as a dice may only have one dot
        clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []

# ------------------------------------------------------------------------------
def overlay_info(frame, dice, blobs):
    """ create overlay text depending on the # of dots and position """

    # Overlay blobs
    for i in blobs:
        position = i.pt
        r = i.size / 2

        cv2.circle(frame, (int(position[0]), int(position[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for j in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(j[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(j[0]),
                    (int(j[1] - textsize[0] / 2),
                     int(j[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

# ------------------------------------------------------------------------------
def record_values(dice):
    """ when called, record currently displayed values """

    events = []
    for i in dice:
       events.append(str(i[0])) 
    return events

# ------------------------------------------------------------------------------
def update_frequencies(dice_number, frequency, events):
    """ updates lists sorting types of dice numbers recorded thus far and
        their frequencies """

    for event in events:
        dice_number.append(event)
        # if not event in dice_number:
        #     dice_number.append(event)
        #     frequency.append(1)
        # else:
        #     indx = dice_number.index(event)
        #     frequency[indx] = frequency[indx] + 1

# ------------------------------------------------------------------------------
def make_plot(dice_numbers):
    """ create a plot of the frequencies of  """

    # create plot with appropreate labeles
    plt.figure()
    plt.title("Dice Histogram")
    plt.xlabel("Dice number")
    plt.ylabel("Frequency")
    # plot values
    plt.hist(dice_numbers, density=False,bins=10)
    plt.show()

# ------------------------------------------------------------------------------
def main():
    dice_numbers = []
    dice_number_frequency = []
    
    # Initialize a video feed
    cap = cv2.VideoCapture(0)

    while(True): # while camera is running

        # Grab the latest image from the video feed
        ret, frame = cap.read()
        blobs = get_blobs(frame)
        
        if len(blobs) > 0:
            dice = get_dice_from_blobs(blobs)
            out_frame = overlay_info(frame, dice, blobs)

        cv2.imshow("frame", frame)

        res = cv2.waitKey(1)

        # record value if "r" is pressed
        if res & 0xFF == ord('r'):
            values = record_values(dice)
            update_frequencies(dice_numbers, dice_number_frequency, values)
            make_plot(dice_numbers)

        # Stop if "q" is pressed
        if res & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

main()