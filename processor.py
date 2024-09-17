import os, sys
import subprocess
import time
from glob import glob

import ants


if __name__ == "__main__":

    '''
    This script performs several functions:
        1) Image conversion (DCM -> NifTI)
        2) Image alignment using antspyx

    It runs asychronously to process DCM frames as soon as they are written to the shared drive.
    '''

    if len(sys.argv) > 1:
        shared_drive_path = sys.argv[1]
        participant = int(sys.argv[2])
    else:
        shared_drive_path = "shared_drive"
        participant = 1


    # Set alignment target
    target_img = ants.image_read(f"{shared_drive_path}/models/sub-{participant:02}/target_img.nii.gz")

    # Watch output directory for dicom files
    source_dir = glob(f"{shared_drive_path}/data/source/*realtime*")
    while not source_dir:
        time.sleep(1)
        source_dir = glob(f"{shared_drive_path}/data/source/*realtime*")
        continue
        
    source_dir = source_dir[0]
    
    print(f"Processing: {source_dir}")

    nifti_dir = os.path.join(shared_drive_path, "data", "nifti")
    aligned_dir = os.path.join(shared_drive_path, "data", "aligned")

    processed_dcms = []
    unprocessed_dcms = []

    t1 = time.time()
    checkpoint = 10

    while(True):

        dicoms = []

        for file in os.listdir(source_dir):
            dicoms.append(file)


        s = set(processed_dcms)
        unprocessed_dcms = [file for file in dicoms if file not in s]

        if (len(unprocessed_dcms) >= 2) or len(processed_dcms) >= 714:
            for file in unprocessed_dcms:
                
                dicom_file = os.path.join(source_dir, file)
                output_filename = f"img_{len(processed_dcms):04}"

                # Construct the dcm2niix command
                command = [
                    'dcm2niix', 
                    '-o', nifti_dir,    # Output directory
                    '-f', output_filename,  # Output filename
                    '-b', 'n',
                    '-v', '0',
                    '-s', 'y', # Single file mode
                    dicom_file            # Input DICOM file
                ]

                # Run the command using subprocess
                dcm2nii = subprocess.call(command, stdout=open(os.devnull, 'wb'))

                # Do registration
                nifti_file = os.path.join(nifti_dir, output_filename + '.nii')
                ants_img = ants.image_read(nifti_file)

                # areg = align_image(ants_img, fmri_ref, target_img, transforms)
                areg = ants.registration( target_img, ants_img, "BOLDAffine" )
                save_filename = os.path.join(aligned_dir, output_filename + ".nii.gz")
                ants.image_write(areg['warpedmovout'], save_filename)

                processed_dcms.append(file)

        aligned_imgs = os.listdir(aligned_dir)

        # Print status intermittently
        t2 = time.time()
        if int(t2 - t1) >= checkpoint:

            print(f"UPDATE: Processed {len(aligned_imgs)} images in {int(t2-t1)} seconds, at a rate of {len(aligned_imgs) / int(t2-t1)} images per second.")

            checkpoint = checkpoint + 10

        # Exit when run is complete
        if len(aligned_imgs) >= 715:
            print("Completed run.")
            exit()
