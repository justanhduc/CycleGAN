# Cycle GAN
A Theano implementation of Cycle GAN

## Requirements

[Theano](http://deeplearning.net/software/theano/)

[Neuralnet](https://github.com/justanhduc/neuralnet)

## Usage

```
python cycleGAN.py path-to-dataset (--out_size 128) (--bs 1) (--n_epochs 200) (--lr 0.0002) (--lambd 10) (--use_sigmoid 0) (--gpu 0)
```

## Results

The following samples were produced after 210k iterations

![monet2photo](https://github.com/justanhduc/CycleGAN/blob/master/samples/monet2photo.png)

![photo2monet](https://github.com/justanhduc/CycleGAN/blob/master/samples/photo2monet.png)

## Credits

The code is a re-implementation of the following paper

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```

Some codes are borrowed from [this Lasagne implentation](https://github.com/tjwei/GANotebooks/blob/master/CycleGAN-lasagne.ipynb). This implementation included some tricks for stability described in the paper, and not included in the Lasagne code.
