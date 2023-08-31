import sys
import subprocess


def stackvids(base_filename, stack_type="hstack", destination=".", num_cams=4):
    if stack_type in ["hstack", "vstack"]:
        if num_cams == 2:
            # os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -filter_complex {stack_type}=inputs=2 {destination}/stack_{base_filename}.mp4')
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    f"{base_filename}_cam0.mp4",
                    "-i",
                    f"{base_filename}_cam1.mp4",
                    "-vcodec",
                    "libx265",
                    "-filter_complex",
                    f"{stack_type}=inputs=2",
                    f"{destination}/stack_{base_filename}.mp4",
                ]
            )
        elif num_cams == 3:
            # os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -i {base_filename}_cam2.mp4 -filter_complex {stack_type}=inputs=3 {destination}/stack_{base_filename}.mp4')
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    f"{base_filename}_cam0.mp4",
                    "-i",
                    f"{base_filename}_cam1.mp4",
                    "-i",
                    f"{base_filename}_cam2.mp4",
                    "-vcodec",
                    "libx265",
                    "-filter_complex",
                    f"{stack_type}=inputs=3",
                    f"{destination}/stack_{base_filename}.mp4",
                ]
            )
        elif num_cams == 4:
            # os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -i {base_filename}_cam2.mp4 -i {base_filename}_cam3.mp4 -filter_complex {stack_type}=inputs=4 {destination}/stack_{base_filename}.mp4')
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    f"{base_filename}_cam0.mp4",
                    "-i",
                    f"{base_filename}_cam1.mp4",
                    "-i",
                    f"{base_filename}_cam2.mp4",
                    "-i",
                    f"{base_filename}_cam3.mp4",
                    "-vcodec",
                    "libx265",
                    "-filter_complex",
                    f"{stack_type}=inputs=4",
                    f"{destination}/stack_{base_filename}.mp4",
                ]
            )
    elif num_cams == 4 and stack_type == "xstack":
        # os.system(f'ffmpeg -i {base_filename}_cam0.mp4 -i {base_filename}_cam1.mp4 -i {base_filename}_cam2.mp4 -i {base_filename}_cam3.mp4 -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" {destination}/stack_{base_filename}.mp4')
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                f"{base_filename}_cam0.mp4",
                "-i",
                f"{base_filename}_cam1.mp4",
                "-i",
                f"{base_filename}_cam2.mp4",
                "-i",
                f"{base_filename}_cam3.mp4",
                "-vcodec",
                "libx265",
                "-filter_complex",
                "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]",
                "-map",
                "[v]",
                f"{destination}/stack_{base_filename}.mp4",
            ]
        )
    else:
        raise Exception(
            "Bad combination of num_cams argument and stack_type argument. Should be 'hstack' or 'vstack' with numcams less than 4."
        )


if __name__ == "__main__":
    # print(list(sys.argv))
    if len(list(sys.argv)) == 2:
        stackvids(sys.argv[1])
    elif len(list(sys.argv)) == 3:
        stackvids(sys.argv[1], stack_type=sys.argv[2])
    elif len(list(sys.argv)) == 4:
        stackvids(sys.argv[1], stack_type=sys.argv[2], destination=sys.argv[3])
    elif len(list(sys.argv)) == 5:
        stackvids(
            sys.argv[1], stack_type=sys.argv[2], destination=sys.argv[3], num_cams=sys.argv[4]
        )
    else:
        raise Exception(
            "enter at least 1 arguments and no more than 4! :) \nInputs: base_filename, stack_type, destination, num_cams"
        )
