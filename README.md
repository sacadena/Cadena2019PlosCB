# Deep convolutional models improve predictions of macaque V1 responses to natural images (Code)
Code for Cadena, S. A, et al. (2019). Deep convolutional models improve predictions of macaque V1 responses to natural images. Plos Computational Biology

## Data License

The data shared with this code is  licensed under a This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. This license requires that you contact us before you use the data in your own research. In particular, this means that you have to ask for permission if you intend to publish a new analysis performed with this data (no derivative works-clause).

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a>

## Setup

To run this code you need the following:
- Python3
- Tensorflow 1.5
- Download the checkpoint weights of the normalized VGG-19 network [here](https://drive.google.com/open?id=1TvVGf2ClDARfSNfjbHLZLTtgHNe_jLVo) (80MB) and store them in the vgg_weights/ folder

<p align="center">
 <img src="Fig3.pdf" width=80%>
</p>

## Citation

If you find our code useful please cite us in your work:

```
@article{cadena2019deep,
  title={Deep convolutional models improve predictions of macaque V1 responses to natural images},
  author={Cadena, Santiago A and Denfield, George H and Walker, Edgar Y and Gatys, Leon A and Tolias, Andreas S and Bethge, Matthias and Ecker, Alexander S},
  journal={Plos Computational Biology},
  year={2019}
}
