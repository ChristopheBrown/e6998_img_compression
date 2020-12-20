# Image Compression with Deep Neural Networks

This repo builds off of the work of Toderici et al. "Full Resolution Image Compression with Recurrent Neural Networks" ([full paper here](https://arxiv.org/pdf/1608.05148.pdf)). Here we have developed a prototype of a deep neural network that seeks to compress 32x32-pixel images to a binarized format smaller than that of the JPEG format. We use the same architecure as the reference paper and, in the linked notebooks conduct a study on how batch size and training set size affects performance. Our report on the project status as of 19 Dec 2020 is in the pdf within the repo.

The code for the GRU compression network is adapted from [this Github repo](https://github.com/zhang-yi-chi/residual-rnn) built in Tensorflow 1.x. Our reposoity is designed to operate in Tensorflow 2.x+.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements.

```bash
pip install -r requirements.txt
```

## Usage

Please see the ```compression_project_gru.ipynb``` and ```compression_project_lstm.ipynb```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
