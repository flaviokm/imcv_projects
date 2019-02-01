import cv2
import argparse
import time

parser = argparse.ArgumentParser(description='Timelapse')
parser.add_argument('--time', metavar='time', nargs='?', const=5, default=5, type=int,
                    help='Time interval between frames')
parser.add_argument('--duration', metavar='duration', nargs='?', const=600, default=600, type=int,
                    help='Duration of filming')
parser.add_argument('--camera', metavar='duration', nargs='?', const='internal', default='internal', type=str,
                    help='Duration of filming')


if __name__ == '__main__':
    parsed = parser.parse_args()
    interval = parsed.time
    duration = parsed.duration
    camera = 0 if parsed.camera == 'internal' else 1
    out_path = './'
    video_cap = cv2.VideoCapture(camera)

    if video_cap.isOpened():
        ret, frame = video_cap.read()
        cv2.imshow('first',frame)
	cv2.waitKey(1)
	if ret:
            height, width, frame_type = frame.shape
            start_time = time.time()
        # fourcc =  cv2.cv.CV_FOURCC(*'MJPG')
            fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
            fps = 12
            video_filename = out_path + 'video_fps%d' % fps + time.ctime()
            video_out = cv2.VideoWriter(
                video_filename + '.avi', fourcc, fps, (width, height))

            fps = 6
            video_filename = out_path + 'video_fps%d' % fps + time.ctime()
            video_out2 = cv2.VideoWriter(
                video_filename + '.avi', fourcc, fps, (width, height))

            cv2.imshow('last_frame', frame)
            key = cv2.waitKey(1)
            while ((time.time()-start_time) < duration) and key & 0xFF != ord('q'):
                ret, frame = video_cap.read()
                video_out.write(frame)
                video_out2.write(frame)
                cv2.imshow('last_frame', frame)
                key = cv2.waitKey(1)
                time.sleep(interval)

            print "timelapse finished \n start_time:%s \t end_time:%s \n lapse_time:%d" % (
                time.ctime(start_time), time.ctime(), interval)
