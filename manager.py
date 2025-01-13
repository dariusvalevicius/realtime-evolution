import os, subprocess
import time
import atexit, signal
import argparse
import shutil
from gooey import Gooey
import json


def clear_temp_files(shared_drive_path):
    '''
    Function to clear temporary realtime data.
    '''

    # Clear previous files
    for folder in ["source", "dicom", "nifti", "aligned"]:

        path = f"{shared_drive_path}/data/{folder}"

        # Create folders if not present
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for root, dirs, files in os.walk(path):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))


@Gooey
def main():

    '''
    This script is a parent for the processor and collector subprocesses.
    It can also optionally activate the MRI simulator script, which copies DCM files from a source folder to the real-time folder on a fixed interval.
    '''

    # Parse incoming arguments
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('participant', type=int, help="Participant number.")
    parser.add_argument('session', type=int, help="Session number.")
    parser.add_argument('run', type=int, help="Run index for this session.")
    parser.add_argument('condition', type=str, choices=['brain', 'ratings'], help="Are brain scores or self-report ratings being used?")

    parser.add_argument('--target', type=str, default="fear", help="Name of target map (e.g., input 'fear' if file is 'fear_map.nii')")
    parser.add_argument('--simulate', action='store_true', help="Is this run being simulated offline?", required=False)
    
    args = parser.parse_args()

    participant = args.participant
    ses = args.session
    run = args.run
    condition = args.condition
    target = args.target

    with open('config.json') as f:
        config = json.load(f)

    shared_drive_path = config["shared_drive_path"]

    output_path = f"{shared_drive_path}/images/sub-{participant:02}/ses-{ses:02}/run-{run}"

    # Clear existing files, if any
    clear_temp_files(shared_drive_path)


    if condition == "brain":
        # Simulation
        if args.simulate:
            sim = subprocess.Popen(["python", "siemens_simulator.py", shared_drive_path])
            atexit.register(os.kill, sim.pid, signal.CTRL_C_EVENT)
            time.sleep(2)

        if condition == "brain":
            # Start processor subprocess
            proc = subprocess.Popen(["python", "processor.py", shared_drive_path, str(participant)])
            atexit.register(os.kill, proc.pid, signal.CTRL_C_EVENT)


    # Start collector subprocess
    collector = subprocess.Popen(['python', 'collector.py', shared_drive_path, output_path, str(participant), target, condition])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    return None

if __name__ == "__main__":
    main()

            



