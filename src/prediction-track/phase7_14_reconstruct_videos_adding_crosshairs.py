# importing libraries
import tsi.tsi_sys as tsis
import tsi.tsi_img_processing as tsiimg
import cv2
import pandas as pd

def get_input_base():
    return 'D:/ou/output/phase7/2_join-ocr-eye-instrument/experiment2/'

def get_output_base():
    return 'D:/ou/output/phase7/14_crosshairs/experiment2/'


if __name__ == "__main__":

    input_files = tsis.list_files(get_input_base())

    tsis.make_dir(get_output_base())

    for input_file in input_files:

        prefix = tsis.get_prefix(input_file)

        if prefix != 'join':

            # read csv
            df = pd.read_csv(input_file)
            print(df)
            # drop the rows with empty name or subject column
            df = df[df['name'].notna()]
            df = df[df['subject'].notna()]

            img_array = []

            # for each row:
            for index, row in df.iterrows():
                # open image
                su = row["subject"]
                name = row["name"]
                filename = 'D:/ou/output/phase0/experiment2/rgb/' + su + '/' + name + '.png'
                img = tsiimg.read_img(filename)
                height, width, layers = img.shape
                if not pd.isna(row["eye_x"]):
                #if isinstance(row["Xloc"], float):
                    eye_x = int(row["eye_x"])
                    eye_y = int(row["eye_y"])
                    cv2.line(img, (eye_x, 0), (eye_x, height), (0, 255, 0), thickness=2)
                    cv2.line(img, (0, eye_y), (width, eye_y), (0, 255, 0), thickness=2)
                if not pd.isna(row["tooltip_x"]):
                    tool_x = int(row["tooltip_x"])
                    tool_y = int(row["tooltip_y"])
                    cv2.line(img, (tool_x, 0), (tool_x, 1060), (255, 0, 0), thickness=2)
                    cv2.line(img, (0, tool_y), (1280, tool_y), (255, 0, 0), thickness=2)
                cv2.imshow("Subject",img)
                img_array.append(img)
                key = cv2.waitKey(10) & 0xff
                # ESC breaks the loop
                if key == 27 : break
                if key == ord("p"): 
                    # Wait until any key press.
                    cv2.waitKey(0)

            size = (width,height)
            fps = 3
            out = cv2.VideoWriter(get_output_base() + prefix + "_crosshairs_" + str(fps) + "fps.mp4", 
                                    cv2.VideoWriter_fourcc(*"DIVX"), fps, size)

            print("Reconstructing video for " + prefix)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
            print("Reconstruction for " + prefix + " done")

        cv2.destroyAllWindows()



    print('*** reconstructing videos completed ***')