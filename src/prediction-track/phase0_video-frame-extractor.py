import tsi.tsi_sys as tsis # abstracts away os and sys file handling (mostly)
import tsi.tsi_img_processing as tsiimg # abstracts away cv2 implementation


def get_input_base():
    return 'd:/ou/input/phase0/'

def get_output_base():
    return 'd:/ou/output/phase0/'

def get_output_folder(prefix, suffix, channels):
    return get_output_base() + suffix + '/' + channels + '/' + prefix + '/'


# takes a video and extracts its frames
def extract_frames(video, prefix, output_path_rgb, output_path_gray):
    
    # initialise counter for the frame names
    frame_no = 0

    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break

        # for greed purposes, store each frame as RGB and as grayscale image
        tsiimg.write_img(output_path_rgb + prefix + '_frame' + str(frame_no) + '.png', frame)
        # convert to grayscale
        frame_gray = tsiimg.convert_to_grayscale(frame)
        tsiimg.write_img(output_path_gray + prefix + '_frame' + str(frame_no) + '.png', frame_gray)
        
        frame_no+=1

def process_video(video_file, suffix):
        # save the prefix of the video file. It is used to name and organise the output frames
        prefix = tsis.get_prefix(video_file, '.')
        output_path_gray = get_output_folder(prefix, suffix, 'gray')
        output_path_rgb = get_output_folder(prefix, suffix, 'rgb')
        tsis.make_dir(output_path_gray)
        tsis.make_dir(output_path_rgb)

        # Create a VideoCapture object and read from input file
        video = tsiimg.capture_video(video_file)

        # Check if video opened successfully
        if not video.isOpened(): 
            print("Error opening video file: " + video_file)
            # this is a less terrible condition. Break and try the next video
            return

        # do the work
        print('Extracting frames of video: ' + video_file)
        extract_frames(video, prefix, output_path_rgb, output_path_gray)

        # when everything done, release the video capture object
        video.release()
        print('Video ' + video_file + ' processed succesfully')

# main driver for 2 levels
# It reads the input folder with the experiment folders (suffix), and creates output folders for both experiments
# For each experiment, it then reads the content (videos) and processes them one by one
# Each video has a prefix, which is used to create a separate folder for each video to store the frames in.
if __name__ == "__main__":

    input_folders = tsis.list_folders(get_input_base())

    for input_folder in input_folders:

        # strip inner folder and prepare output folder
        suffix = tsis.get_basename(input_folder)

        # collect the videos
        video_files = tsis.list_files(input_folder)

        for video_file in video_files:

            process_video(video_file, suffix)
    
    print('*** Frame extraction completed ***')