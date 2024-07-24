import os, sys
import subprocess
import time

import ants


if __name__ == "__main__":

    if len(sys.argv) > 1:
        shared_drive_path = sys.argv[1]
        participant = int(sys.argv[2])
    else:
        shared_drive_path = "shared_drive"
        participant = 1


    target_img = ants.image_read(f"{shared_drive_path}/models/sub-{participant:02}/target_img.nii.gz")

    # Watch output directory for dicom files
    dicom_dir = os.path.join(shared_drive_path, "data", "dicom")#"data/dicom"
    nifti_dir = os.path.join(shared_drive_path, "data", "nifti")
    aligned_dir = os.path.join(shared_drive_path, "data", "aligned")


    # dicoms = []
    processed_dcms = []
    unprocessed_dcms = []

    t1 = time.time()
    checkpoint = 10

    while(True):

        dicoms = []

        for folder in os.listdir(dicom_dir):
            contents = os.listdir(os.path.join(dicom_dir, folder))
            if contents:
                dicoms.append(folder)


        s = set(processed_dcms)
        unprocessed_dcms = [file for file in dicoms if file not in s]

        # niftis_to_process = []

        if unprocessed_dcms:
            # print(unprocessed_dcms)
            for folder in unprocessed_dcms:

                # dicom_file = os.listdir(os.path.join(dicom_dir, folder))
                path = os.path.join(dicom_dir, folder)
                dicom_file = os.path.join(path, os.listdir(path)[0])
                # print(dicom_file)

                output_filename = f"img_{len(processed_dcms):04}"
                # output_filename = Path(dicom_file).stem
                # print(output_filename)

                # Construct the dcm2niix command
                command = [
                    'dcm2niix', 
                    '-o', nifti_dir,    # Output directory
                    '-f', output_filename,  # Output filename
                    '-b', 'n',
                    '-v', '0',
                    dicom_file            # Input DICOM file
                ]

                # Run the command using subprocess
                dcm2nii = subprocess.Popen(command)
                dcm2nii.wait()

                nifti_file = os.path.join(nifti_dir, output_filename + '.nii')

                # while not os.path.isfile(nifti_file):
                #     pass

                # Do registration
                ants_img = ants.image_read(nifti_file)

                # areg = align_image(ants_img, fmri_ref, target_img, transforms)
                areg = ants.registration( target_img, ants_img, "BOLDAffine" )

                save_filename = os.path.join(aligned_dir, output_filename + ".nii.gz")

                ants.image_write(areg['warpedmovout'], save_filename)

                processed_dcms.append(folder)


        t2 = time.time()
        if int(t2 - t1) >= checkpoint:

            aligned_imgs = os.listdir(aligned_dir)

            print(f"UPDATE: Processed {len(aligned_imgs)} images in {int(t2-t1)} seconds, at a rate of {len(aligned_imgs) / int(t2-t1)} images per second.")

            checkpoint = checkpoint + 10

            if len(aligned_imgs) == 1025:
                time.sleep(5)
                exit()


        # if len(processed_dcms) == 1025:
        #     time.sleep(5)
        #     exit()

