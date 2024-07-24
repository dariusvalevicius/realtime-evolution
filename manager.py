import os, subprocess, sys
import time
import atexit, signal


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # output_path = sys.argv[1]
        shared_drive_path = sys.argv[1]
        participant = int(sys.argv[2])
        run = int(sys.argv[3])
    else:
        # output_path = "shared_drive/images/sub-01/run-1"
        shared_drive_path = "shared_drive"
        participant = 1
        run = 1

    output_path = f"{shared_drive_path}/images/sub-{participant:02}/run-{run}"

    ## for testing:
    sim = subprocess.Popen(["python", "siemens_simulator.py", shared_drive_path])
    atexit.register(os.kill, sim.pid, signal.CTRL_C_EVENT)

    time.sleep(2)

    # # Start converter
    # conv = subprocess.Popen(["python", "converter.py"])
    # atexit.register(os.kill, conv.pid, signal.CTRL_C_EVENT)

    # # Start aligner
    # align = subprocess.Popen(["python", "aligner.py", str(participant)])
    # atexit.register(os.kill, align.pid, signal.CTRL_C_EVENT)

    proc = subprocess.Popen(["python", "processor.py", shared_drive_path, str(participant)])
    atexit.register(os.kill, proc.pid, signal.CTRL_C_EVENT)

    # Start image generator subprocess
    generator = subprocess.Popen(['python', 'generator.py', output_path])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(os.kill, generator.pid, signal.CTRL_C_EVENT)

    # Start evaluator subprocess
    evaluator = subprocess.Popen(['python', 'evaluator.py', shared_drive_path, output_path, str(participant)])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(os.kill, evaluator.pid, signal.CTRL_C_EVENT)

    while(True):
        pass



