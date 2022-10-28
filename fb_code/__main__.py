# Runs rest of code
import demographics
import feature_extraction

if __name__ == '__main__':

    # Extract features
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        make_patient_data(patient)

    combine_files(subject_ids)
    print('Processing complete.')

    # Join with patient demographics
    rp = rparser()