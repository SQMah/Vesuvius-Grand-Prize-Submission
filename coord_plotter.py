import cv2

if __name__ == "__main__":

    img_path = "./test.png"
    img = cv2.imread(img_path)

    annotations = {
        'Alpha': {
            'boxes': [((200, 60), (240, 80)), ((1080, 130), (1120, 150)), ((70, 290), (100, 310)),
                      ((2390, 80), (2420, 100))]
        },
        'Beta': {
            'boxes': [((390, 140), (420, 160)), ((1700, 190), (1730, 210))]
        },
        'Gamma': {
            'boxes': [((620, 80), (660, 100)), ((120, 130), (160, 150)), ((2590, 90), (2620, 110))]
        },
        'Delta': {
            'boxes': [((800, 250), (830, 270)), ((1450, 70), (1480, 90))]
        },
        'Epsilon': {
            'boxes': [((980, 55), (1010, 75)), ((2100, 320), (2130, 340))]
        },
        'Zeta': {
            'boxes': [((1600, 155), (1630, 175))]
        },
        'Eta': {
            'boxes': [((1150, 95), (1180, 115)), ((2500, 200), (2530, 220))]
        },
        'Theta': {
            'boxes': [((540, 45), (570, 65)), ((2000, 270), (2030, 290))]
        },
        'Iota': {
            'boxes': [((920, 100), (950, 120)), ((2300, 140), (2330, 160))]
        },
        'Kappa': {
            'boxes': [((300, 290), (330, 310)), ((2650, 300), (2680, 320))]
        },
        'Lambda': {
            'boxes': [((850, 100), (880, 120)), ((1900, 200), (1930, 220))]
        },
        # ... add more annotations as needed
    }

    # For each annotation, draw the bounding boxes and the English letter name
    for letter, data in annotations.items():
        for box in data['boxes']:
            cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)  # Draw rectangle
            # Display the English name slightly above the bounding box
            cv2.putText(img, letter, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)

    # Display the annotated image
    cv2.imshow('Annotated Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
