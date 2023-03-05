import  tensorflow as tf
''' 2.6.2'''
print( "Tensorflow version : " , tf.__version__)


# from google.cloud import storage
#
# # Initialise a client
# storage_client = storage.Client("[Your project name here]")
# # Create a bucket object for our bucket
# bucket = storage_client.get_bucket("gs://gresearch/lens-flare")
# # Create a blob object from the filepath
# blob = bucket.blob("folder_one")
# # Download the file to a destination
# blob.download_to_filename("data")