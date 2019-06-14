import cv2
import dlib
import numpy as np

from imutils import face_utils
from collections import OrderedDict

capture = cv2.VideoCapture(0)

facial_feature_coordinates = {}

sp = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sp)

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def shape_to_np_array(shape, dtype="int"):
    coordinates = np.zeros((68,2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i))

    return coordinates

def facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        facial_feature_coordinates[name] = pts

        if name == "jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    print(facial_feature_coordinates)
    return output

while True:
    ret, image = capture.read()
    dlib.shape_predictor()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        output = facial_landmarks(image, shape)

        cv2.imshow("Video", output)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()