import os, subprocess
import time
import atexit, signal
import argparse
import shutil


def clear_temp_files(shared_drive_path):
    '''
    Function to clear temporary realtime data.
    '''

    # Clear previous files
    for folder in ["source", "dicom", "nifti", "aligned"]:
        for root, dirs, files in os.walk(f"{shared_drive_path}/data/{folder}"):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


if __name__ == "__main__":

    '''
    This script is a parent for the processor and collector subprocesses.
    It can also optionally activate the MRI simulator script, which copies DCM files from a source folder to the real-time folder on a fixed interval.
    '''

    # Parse incoming arguments
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('participant', type=int, help="Participant number.")
    parser.add_argument('session', type=int, help="Real-time session number.")
    parser.add_argument('run', type=int, help="Run index for this session.")
    parser.add_argument('target', type=str, help="Evolution target. Example options:\n'fear'\n'cute'\n'disgust'\n'neurosynth'")
    parser.add_argument('--simulate', action='store_true', help="Is this run being simulated offline?", required=False)
    parser.add_argument('--shared_drive_path', type=str, help="Alternative shared drive path.", required=False)
    parser.add_argument('--diffusion_steps', type=int, default=25, help="Number of diffusion steps per image", required=False)

    
    args = parser.parse_args()

    participant = args.participant
    ses = args.session
    run = args.run
    target = args.target
    diffusion_steps = args.diffusion_steps

    # Set paths
    if args.shared_drive_path:
        shared_drive_path = args.shared_drive_path
    else:
        shared_drive_path = r"C:\Users\TD_Lab\Desktop\Realtime\shared_drive"

    output_path = f"{shared_drive_path}/images/sub-{participant:02}/ses-{ses:02}/run-{run}"


    # Clear existing files, if any
    clear_temp_files(shared_drive_path)


    # Simulator
    if args.simulate:
        sim = subprocess.Popen(["python", "siemens_simulator.py", shared_drive_path])
        atexit.register(os.kill, sim.pid, signal.CTRL_C_EVENT)
        time.sleep(2)


    # Start processor subprocess
    proc = subprocess.Popen(["python", "processor.py", shared_drive_path, str(participant)])
    atexit.register(os.kill, proc.pid, signal.CTRL_C_EVENT)
    

    # Start collector subprocess
    collector = subprocess.Popen(['python', 'collector.py', shared_drive_path, output_path, str(participant), target, str(diffusion_steps)])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(os.kill, collector.pid, signal.CTRL_C_EVENT)

    while(True):
    
        poll = collector.poll()
        if poll is None:
            time.sleep(1)
        else:
            print("Collector closed.")
            # Clear existing files, if any
            clear_temp_files(shared_drive_path)
            exit()
            



