import cv2
import numpy as np
import math
import paho.mqtt.client as mqtt
import struct
import create_grid


def axisangle2euler(rotv):
    """
    Convert the axis-angle representation of a rotation to euler angles (roll, pitch, yaw).

    rotv: 3d vector whose direction represents the rotation axis and length (norm) the angle in radians
    """
    angle = np.linalg.norm(rotv)

    if angle == 0:
        return 0, 0, 0

    # Normalize the vector
    rotv = rotv / angle

    x = rotv[0]
    y = rotv[1]
    z = rotv[2]

    # roll = math.atan2(x * math.sin(angle) - y * z * (1 - math.cos(angle)), 1 - (x * x + z * z) * (1 - math.cos(angle)))
    # pitch = math.asin(x * y * (1 - math.cos(angle)) + z * math.sin(angle))
    # yaw = math.atan2(y * math.sin(angle) - x * z * (1 - math.cos(angle)), 1 - (y * y + z * z) * (1 - math.cos(angle)))

    # Create the basis unit vectors
    eyes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    # Calculate rotation matrix from the
    rotm = np.zeros((3, 0))
    for e in eyes:
        v = e * math.cos(angle) + np.cross(rotv, e) * math.sin(angle) + rotv * np.dot(rotv, e) * (1 - math.cos(angle))
        rotm = np.concatenate((rotm, v[np.newaxis].T), axis=1)

    # Create RPY angles from rotation matrix
    roll = math.atan2(rotm[2, 1], rotm[2, 2])
    pitch = math.atan2(-rotm[2, 0], math.sqrt(rotm[2, 1] ** 2 + rotm[2, 2] ** 2))
    yaw = math.atan2(rotm[1, 0], rotm[0, 0])

    return roll, pitch, yaw


def wrapTo2pi(angle):
    new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    if new_angle < 0:
        new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
    return new_angle


def camera2marker_plane(vec3d):
    pass


mqtt_connected = False
def on_mqtt_connected(client, userdata, flags, rc):
    global mqtt_connected
    mqtt_connected = True

def on_mqtt_disconnected(client, userdata, flags, rc):
    global mqtt_connected
    mqtt_connected = False

if __name__ == "__main__":
    WINDOW_NAME = "Camera feed"

    # ArUco marker parameters
    markerNum = 10
    robot_id = 0
    markerSize = 4
    markerLength = 0.1  # [m]
    camera_height = 2.54 # [m]

    # Set up MQTT client
    mqttClient = mqtt.Client("CameraPosSys")
    mqttClient.username_pw_set('omnibot', 'myomni')
    mqttClient.on_connect = on_mqtt_connected
    mqttClient.on_disconnect = on_mqtt_disconnected

    mqttClient.connect('localhost')
    mqttClient.loop_start()

    # Camera distortion parameters: https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html
    # intrinsic matrix: [ f_x,   0, c_x;
    #                       0, f_y, c_y;
    #                       0,   0,   1]
    #
    # distortion coeffs: [k1, k2, p1, p2<, k3, <k4,k5,k6>, <s1,s2,s3,s4>>>] k = radial, p = tangential

    # Set up camera (LEGO Logitech C270)
    # intrinsicMatrix = np.array(
    #     [[1.512615778711343e+03, 0, 4.520537720295670e+02],
    #      [0, 1.486977356136727e+03, 4.308354027197286e+02],
    #      [0, 0, 1]])
    # distortionCoeff = np.array(
    #     [-0.169723383281475, 0.897620109447660, -0.019083126489556, -0.031127548033969, -1.294627505010236])

    # Set up camera (Samsung Galaxy A8. droidcam)
    # intrinsicMatrix = np.array(
    #     [[1.133249927037626e+03, 0.243444325232128, 9.572915390341526e+02],
    #      [0, 1.136035808896597e+03, 5.202743831457274e+02],
    #      [0,    0,  1]])
    #
    # distortionCoeff = np.array(
    #     [0.139126078793047,	-0.443951971076217, -0.001849431061725, 2.027605013930054e-04, 0.366009652199775])

    # Set up camera (Samsung Galaxy A8. Iriun)
    intrinsicMatrix = np.array(
        [[1.577768278272033e+03, -4.160135230572470, 9.714924733903853e+02],
         [0, 1.571801213631334e+03, 5.179746209450728e+02],
         [0, 0, 1]])

    distortionCoeff = np.array(
        [0.217876724428416, -0.997854302784941, -0.002082456736448, 7.395556748261536e-04, 1.526559272031024])

    # Set up OpenCV for reading and displaying camera images
    cv2.namedWindow(WINDOW_NAME)
    camCapture = cv2.VideoCapture(1)
    camCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    position_log = np.zeros((0,3))
    world_frame_log = np.zeros((0,7))

    cameraAvailable = True
    # if camCapture.isOpened(): # try to get the first frame
    #     cameraAvailable, frame = camCapture.read()
    # else:
    #     cameraAvailable = False

    # Set up OpenCV for ArUco marker detection
    myArucoDict = cv2.aruco.custom_dictionary(markerNum, markerSize, 0)
    arucoParams = cv2.aruco.DetectorParameters_create()

    # Initialize vectors and transformation
    trvec_World = None
    trvec_World_cov = None
    T_w2c_hom = None
    T_c2w_hom = None
    process_noise = np.eye(7) * 1e-5
    meas_cov = np.eye(7)

    # Main loop
    while True:
        cameraAvailable, frame = camCapture.read()

        if cameraAvailable:

            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if not mqtt_connected:
                mqttClient.connect('localhost')

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, myArucoDict, parameters=arucoParams)
            if ids is not None:
                for corner, id in zip(corners, ids):
                    if id is not None:

                        if id == 0:
                            markerLength = 0.1
                        elif id == 1:
                            markerLength = 0.195

                        # rvecs: angle axis representation, tvecs:translation
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, markerLength, intrinsicMatrix,
                                                                              distortionCoeff)

                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        cv2.aruco.drawAxis(frame, intrinsicMatrix, distortionCoeff, rvecs[0], tvecs[0], markerLength)

                        if id == 1:
                            if trvec_World is None:
                                # initialization of Kalman filter
                                rvec_norm = np.linalg.norm(rvecs[0])
                                rvec = rvecs[0]/rvec_norm
                                trvec_World = np.vstack((tvecs[0].T, rvec.T, rvec_norm))
                                trvec_World_cov = meas_cov
                            else:
                                # correct for +/-pi jumping
                                rvec_norm = np.linalg.norm(rvecs[0])
                                rvec = rvecs[0].T/rvec_norm

                                sign_corr = np.sign(rvec.T @ trvec_World[3:-1])

                                rvec = sign_corr * rvec
                                rvec_norm = wrapTo2pi(sign_corr * rvec_norm)

                                trvec_new = np.vstack((tvecs[0].T, rvec, rvec_norm))

                                world_frame_log = np.vstack((world_frame_log, trvec_World.T))

                                # Kalman filter with static "dynamics"
                                K = trvec_World_cov @ np.linalg.inv(trvec_World_cov + meas_cov)
                                trvec_World = trvec_World + K @ (trvec_new - trvec_World)

                                trvec_World_cov = (np.eye(7) - K) @ trvec_World_cov @ (np.eye(7) - K).T + K @ meas_cov @ K.T
                                trvec_World_cov = trvec_World_cov + process_noise

                            R_w2c, _ = cv2.Rodrigues(trvec_World[3:-1]*trvec_World[-1])
                            T_w2c_hom = np.vstack((np.hstack((R_w2c, trvec_World[0:3])), np.array([0, 0, 0, 1])))   # base of world frame, expressed in camera frame
                            T_c2w_hom = np.linalg.inv(T_w2c_hom)

                            x_grid = 10
                            y_grid = 8
                            grid = create_grid.create_grid((x_grid, y_grid), (0.25, 0.25), T_w2c_hom, intrinsicMatrix,
                                                           distortionCoeff)
                            for y_idx in range(y_grid):
                                for x_idx in range(x_grid):
                                    from_p = grid[y_idx * x_grid + x_idx, :]
                                    if x_idx < x_grid - 1:
                                        to_p = grid[y_idx * x_grid + x_idx + 1, :]
                                        cv2.line(frame, from_p.astype(int), to_p.astype(int), [0, 255, 0], 3)

                                    if y_idx < y_grid - 1:
                                        to_p = grid[(y_idx + 1) * x_grid + x_idx, :]
                                        cv2.line(frame, from_p.astype(int), to_p.astype(int), [0, 255, 0], 3)

                        elif id == 0:
                            if T_w2c_hom is None:
                                cv2.putText(frame, f'No world frame detected!', (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                                            3, cv2.LINE_8)
                                continue

                            R_r2c, _ = cv2.Rodrigues(rvecs[0])
                            # rotm_cam2robot = rotm_cam2robot

                            r, p, y = axisangle2euler(rvecs[0][0])

                            x_pos_kamera = tvecs[0][0][0]
                            y_pos_kamera = tvecs[0][0][1]
                            z_pos_kamera = tvecs[0][0][2]

                            robot_posMeas_camera = np.array([[x_pos_kamera], [y_pos_kamera], [z_pos_kamera]])

                            # Create projection matrix
                            # V = span(v1, v2) az alter amire vetitunk
                            v1_cam = T_w2c_hom[0:3, 0:1]
                            v2_cam = T_w2c_hom[0:3, 1:2]
                            # W = span(w1) az alter ami menten vetitunk
                            w1_cam = T_w2c_hom[0:3, 2:3]

                            # Projection matrix (src: LinAlg/EA04)
                            P = np.hstack([v1_cam, v2_cam, np.zeros((3, 1))]) @ np.linalg.inv(np.hstack([v1_cam, v2_cam, w1_cam]))

                            # # Create base of the world coordinate system
                            # b1_world_camera = P @ np.array([[1], [0], [0]])
                            # # b2_world_camera = P @ np.array([[0], [-1], [0]])
                            #
                            # b1_world_camera = b1_world_camera/np.linalg.norm(b1_world_camera)
                            # # b2_world_camera = b2_world_camera / np.linalg.norm(b2_world_camera) #  PROBLÉMA: nem biztos, hogy ortogonális  a két tengely
                            # b3_world_camera = np.cross(v1_cam, v2_cam, axis=0)
                            # b2_world_camera = np.cross(b3_world_camera, b1_world_camera, axis=0)
                            #
                            # B_world_camera = np.hstack([b1_world_camera, b2_world_camera, b3_world_camera])
                            # R_world2camera = B_world_camera
                            #
                            # T_w2c_hom = np.vstack((np.hstack((R_world2camera, np.array([[0,0,camera_height]]).T)), np.array([0,0,0,1])))


                            # Transform robot position to world coordinates
                            robot_posProj_camera = P @ robot_posMeas_camera
                            robot_posProj_world = T_c2w_hom @ np.vstack((robot_posProj_camera, np.array([1])))

                            xdir_robot_world = T_c2w_hom[0:3,0:3] @ R_r2c[:, 0:1]

                            # pose estimation returns marker to camera transformation, but we need camera to marker
                            # x_pos = -tvecs[0][0][0]
                            # y_pos = -tvecs[0][0][1]
                            # heading = -y

                            x_pos = robot_posProj_world[0] / robot_posProj_world[3]
                            y_pos = robot_posProj_world[1] / robot_posProj_world[3]
                            z_pos = robot_posProj_world[2] / robot_posProj_world[3]

                            heading = float(np.arctan2(xdir_robot_world[1], xdir_robot_world[0]))

                            # TODO: modify so ID is coded in the topic
                            payload = struct.pack('<BBfff', 12,0x93,  x_pos, y_pos, heading)
                            mqttClient.publish('cps', payload)

                            position_log = np.vstack((position_log, np.array([x_pos_kamera, y_pos_kamera, z_pos_kamera])))

                            cv2.putText(frame, f'X: {x_pos}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                                        3, cv2.LINE_8)
                            cv2.putText(frame, f'Y: {y_pos}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                                        3, cv2.LINE_8)
                            cv2.putText(frame, f'Z: {z_pos}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                                        3, cv2.LINE_8)
                            cv2.putText(frame, f'H: {np.rad2deg(heading)}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                                        3, cv2.LINE_8)
                            # print(f'State: {x_pos}\t{y_pos}\t{heading}')


            cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            # np.savetxt("pos_log.csv",position_log, delimiter=",")
            # np.savetxt("world_frame_log.csv", world_frame_log, delimiter=",")
            break

    camCapture.release()
    cv2.destroyWindow(WINDOW_NAME)
