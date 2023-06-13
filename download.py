import urllib.request
import os

######### Downlaod the dataset ############
# Create the data subdirectory if not there.
if not os.path.exists('./data/'):
    os.mkdir('./data/')

filename = './data/attack_annotated_comments.tsv'

# Download download if we already have it!
if not os.path.exists(filename):
    # URL for the CSV file (~55.4MB) containing the wikipedia comments.
    url = 'https://ndownloader.figshare.com/files/7554634'

    # Download the dataset.
    print('Downloading Wikipedia Attack Comments dataset (~55.4MB)...')
    urllib.request.urlretrieve(url, filename)

    print('  DONE.')


######### Download Embeddings #############
# Create the data subdirectory if not there.
if not os.path.exists('./data/'):
    os.mkdir('./data/')

files = [
    ('./data/questions-words.txt', 'http://download.tensorflow.org/data/questions-words.txt'),
]

for (filename, url) in files:
    # Download download if we already have it!
    if not os.path.exists(filename):

        # Download the dataset.
        print('Downloading', filename)

        urllib.request.urlretrieve(url, filename)

        print('  DONE.')

    else:
        print('Skipping', filename)