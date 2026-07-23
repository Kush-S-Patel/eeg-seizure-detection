import boto3

s3 = boto3.client("s3")

bucket = "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"

key = "EEG/bids/Neurotech/sub-Neurotech390/ses-1/eeg/sub-Neurotech390_ses-1_task-EEG_eeg.json"

s3.download_file(bucket, key, "test.json")

print("done")