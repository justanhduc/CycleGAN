import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import theano
from theano import tensor as T
from scipy import misc
from random import sample, shuffle, randint
import time

import neuralnet as nn


def reflect_pad(x, width, batch_ndim=1):
    """
    Pad a tensor with a constant value.
    Parameters
    ----------
    x : tensor
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """

    # Idea for how to make this happen: Flip the tensor horizontally to grab horizontal values, then vertically to grab vertical values
    # alternatively, just slice correctly
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    # Create output array
    out = T.zeros(output_shape)

    # Vertical Reflections
    out = T.set_subtensor(out[:, :, :width, width:-width],
                          x[:, :, width:0:-1, :])  # out[:,:,:width,width:-width] = x[:,:,width:0:-1,:]
    out = T.set_subtensor(out[:, :, -width:, width:-width],
                          x[:, :, -2:-(2 + width):-1, :])  # out[:,:,-width:,width:-width] = x[:,:,-2:-(2+width):-1,:]

    # Place X in out
    # out = T.set_subtensor(out[tuple(indices)], x) # or, alternative, out[width:-width,width:-width] = x
    out = T.set_subtensor(out[:, :, width:-width, width:-width], x)  # out[:,:,width:-width,width:-width] = x

    # Horizontal reflections
    out = T.set_subtensor(out[:, :, :, :width],
                          out[:, :, :, (2 * width):width:-1])  # out[:,:,:,:width] = out[:,:,:,(2*width):width:-1]
    out = T.set_subtensor(out[:, :, :, -width:], out[:, :, :, -(width + 2):-(
                2 * width + 2):-1])  # out[:,:,:,-width:] = out[:,:,:,-(width+2):-(2*width+2):-1]
    return out


class ReflectLayer(nn.Layer):

    def __init__(self, input_shape, width, batch_ndim=2, layer_name='ReflectLayer'):
        super(ReflectLayer, self).__init__(input_shape, layer_name)
        self.width = width
        self.batch_ndim = batch_ndim

    @property
    def output_shape(self):
        output_shape = list(self.input_shape)

        if isinstance(self.width, int):
            widths = [self.width] * (len(self.input_shape) - self.batch_ndim)
        else:
            widths = self.width

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def get_output(self, input):
        return reflect_pad(input, self.width, self.batch_ndim)


def ReflectPaddingConv(input_shape, num_filters, filter_size=3, stride=1, activation='relu', use_batchnorm=True,
                       layer_name='ReflectPaddingConv', **kwargs):
    assert filter_size % 2 == 1
    pad_size = filter_size >> 1
    block = nn.Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(ReflectLayer(block.output_shape, pad_size, layer_name=layer_name+'/Reflect'))
    if use_batchnorm:
        block.append(
            nn.ConvNormAct(block.output_shape, num_filters, filter_size, nn.Normal(.02), border_mode=0, stride=stride,
                           activation=activation, layer_name=layer_name + '/conv_bn_act', normalization='gn',
                           groups=num_filters))
    else:
        block.append(nn.ConvolutionalLayer(block.output_shape, num_filters, filter_size, nn.Normal(.02), border_mode=0,
                                           stride=stride, activation=activation, layer_name=layer_name+'/conv'))
    return block


def resblock(input_shape, num_filters, block_name, **kwargs):
    block = nn.Sequential(input_shape=input_shape, layer_name=block_name)
    block.append(ReflectPaddingConv(input_shape, num_filters, layer_name=block_name + '/conv1'))
    block.append(ReflectPaddingConv(input_shape, num_filters, activation='linear', layer_name=block_name + '/conv2'))
    return block


class ResNetGenerator(nn.Sequential):
    def __init__(self, input_shape, num_filters=64, name='ResNetGen'):
        super(ResNetGenerator, self).__init__(input_shape=input_shape, layer_name=name)

        self.append(ReflectPaddingConv(input_shape, num_filters, 7, layer_name=name + '/first'))
        for m in (2, 4):
            self.append(
                nn.ConvNormAct(self.output_shape, num_filters * m, 4, stride=2, border_mode=1, init=nn.Normal(.02),
                               layer_name=name + '/conv_%d' % (num_filters * m), normalization='gn',
                               groups=num_filters * m))

        for i in range(6):
            self.append(nn.ResNetBlock(self.output_shape, num_filters * 4, block=resblock, layer_name=name+'/resblock_%d' % i))

        for m in (2, 1):
            shape = tuple(i * 2 for i in self.output_shape[2:])
            self.append(nn.TransposedConvolutionalLayer(self.output_shape, num_filters * m, 3, shape, nn.Normal(.02),
                                                        layer_name=name+'/transconv_%d' % (num_filters * m), activation='linear'))
            self.append(nn.GroupNormLayer(self.output_shape, name + '/bn_%d' % m, groups=self.output_shape[1],
                                          activation='relu'))

        self.append(
            ReflectPaddingConv(self.output_shape, 3, 7, activation='tanh', use_batchnorm=False, layer_name=name + '/output'))


class Discriminator(nn.Sequential):
    def __init__(self, input_shape, num_filters, max_layers=3, use_sigmoid=True, name='Discriminator'):
        super(Discriminator, self).__init__(input_shape=input_shape, layer_name=name)

        self.append(nn.ConvolutionalLayer(self.output_shape, num_filters, 4, nn.Normal(.02), border_mode=1, stride=2,
                                          activation='lrelu', layer_name=name + '/conv_0', alpha=.2))
        for layer_num in range(1, max_layers):
            n_filters = num_filters * min(2**layer_num, 8)
            self.append(nn.ConvNormAct(self.output_shape, n_filters, 4, nn.Normal(.02), border_mode=1, stride=2,
                                       activation='lrelu', layer_name=name + '/conv_%d' % layer_num,
                                       normalization='gn', groups=n_filters, alpha=.2))

        n_filters = num_filters * min(2**max_layers, 8)
        self.append(
            nn.ConvNormAct(self.output_shape, n_filters, 4, nn.Normal(.02), border_mode=1, stride=1, activation='lrelu',
                           layer_name=name + '/conv_%d' % max_layers, normalization='gn',
                           groups=n_filters, alpha=.2))

        self.append(nn.ConvolutionalLayer(self.output_shape, 1, 4, nn.Normal(.02), border_mode=1, stride=1,
                                          activation='sigmoid' if use_sigmoid else 'linear',
                                          layer_name=name + '/output'))


class DataManager(nn.DataManager):
    def __init__(self, output_size, placeholders, path, batchsize, n_epochs, shuffle, type):
        super(DataManager, self).__init__(None, placeholders, path=path, batch_size=batchsize, n_epochs=n_epochs,
                                          shuffle=shuffle)
        self.type = type
        self.output_size = output_size
        self.load_data()

    def load_data(self):
        self.dataset = [os.listdir(self.path + '/trainA'),
                        os.listdir(self.path + '/trainB')] if self.type == 'train' else [
            os.listdir(self.path + '/testA'), os.listdir(self.path + '/testB')]
        self.dataset = [[self.path + '/trainA/' + file for file in self.dataset[0]],
                        [self.path + '/trainB/' + file for file in self.dataset[1]]] if self.type == 'train' else [
            [self.path + '/testA/' + file for file in self.dataset[0]],
            [self.path + '/testB/' + file for file in self.dataset[1]]]
        self.data_size = max(len(self.dataset[0]), len(self.dataset[1])) if self.type == 'train' \
            else min(len(self.dataset[0]), len(self.dataset[1]))

    def generator(self):
        num_batches = self.data_size // self.batch_size
        datasetA = list(self.dataset[0]) if len(self.dataset[0]) > len(self.dataset[1]) else list(self.dataset[1])
        datasetB = list(self.dataset[1]) if len(self.dataset[0]) > len(self.dataset[1]) else list(self.dataset[0])
        if self.type == 'train':
            ratio = self.data_size // len(datasetB)
            datasetB = sample(datasetB, self.data_size - len(datasetB)*ratio) + datasetB*ratio
        else:
            datasetA = datasetA[:self.data_size]

        if self.shuffle:
            shuffle(datasetA)
            shuffle(datasetB)
        for i in range(num_batches):
            batch = (datasetA[i * self.batch_size:(i + 1) * self.batch_size],
                     datasetB[i * self.batch_size:(i + 1) * self.batch_size])
            batch = (np.array(
                [misc.imresize(misc.imread(file), (self.output_size, self.output_size)) for file in batch[0]], 'float32'),
                     np.array(
                [misc.imresize(misc.imread(file), (self.output_size, self.output_size)) for file in batch[1]], 'float32'))
            batch = (np.transpose(batch[0], (0, 3, 1, 2)) / 127.5 - 1.,
                     np.transpose(batch[1], (0, 3, 1, 2)) / 127.5 - 1.)
            yield batch

    def unnormalize(self, input):
        return input / 2. + .5


def train(image_shape, path, bs=1, lambd=10., lr=2e-4, beta1=.5, n_epochs=200):
    gen_x2y = ResNetGenerator((None,)+image_shape, name='Generator_X2Y')
    gen_y2x = ResNetGenerator((None,)+image_shape, name='Generator_Y2X')
    dis_x = Discriminator((None,)+image_shape, 64, name='DiscriminatorX', use_sigmoid=False)
    dis_y = Discriminator((None,)+image_shape, 64, name='DiscriminatorY', use_sigmoid=False)

    X = T.tensor4('inputX')
    Y = T.tensor4('inputY')
    X_fake_t = T.tensor4('inputX_fake')
    Y_fake_t = T.tensor4('inputY_fake')
    X_ = theano.shared(np.zeros((bs,)+image_shape, 'float32'), 'inputX placeholder')
    Y_ = theano.shared(np.zeros((bs,)+image_shape, 'float32'), 'inputY placeholder')
    lr_ = theano.shared(np.float32(lr), 'learning rate')

    nn.set_training_status(True)
    Y_fake = gen_x2y(X)
    X_fake = gen_y2x(Y)

    pred_X_real = dis_x(X)
    pred_X_fake = dis_x(X_fake_t)

    pred_Y_real = dis_y(Y)
    pred_Y_fake = dis_y(Y_fake_t)

    X_from_Y_fake = gen_y2x(Y_fake)
    Y_from_X_fake = gen_x2y(X_fake)

    dis_X_loss = T.mean((pred_X_real - 1.) ** 2.) + T.mean(pred_X_fake ** 2.)
    dis_Y_loss = T.mean((pred_Y_real - 1.) ** 2.) + T.mean(pred_Y_fake ** 2.)
    gen_X_loss = T.mean((dis_x(X_fake) - 1.) ** 2.)
    gen_Y_loss = T.mean((dis_y(Y_fake) - 1.) ** 2.)
    cycle_loss = nn.norm_error(X_from_Y_fake, X, 1) + nn.norm_error(Y_from_X_fake, Y, 1)

    loss_gen = gen_X_loss + gen_Y_loss + cycle_loss * lambd
    loss_dis = dis_X_loss + dis_Y_loss
    updates_gen = nn.adam(loss_gen, gen_x2y.trainable+gen_y2x.trainable, lr_, beta1)
    updates_dis = nn.adam(loss_dis, dis_x.trainable+dis_y.trainable, lr_, beta1)
    train_gen = nn.function([], [X_fake, Y_fake, gen_X_loss, gen_Y_loss, cycle_loss], updates=updates_gen,
                            givens={X: X_, Y: Y_}, name='train generators')
    train_dis = nn.function([X_fake_t, Y_fake_t], [dis_X_loss, dis_Y_loss], updates=updates_dis, givens={X: X_, Y: Y_},
                            name='train discriminators')

    nn.set_training_status(False)
    generate_X = nn.function([], X_fake, givens={Y: Y_}, name='generate X from Y')
    generate_Y = nn.function([], Y_fake, givens={X: X_}, name='generate Y from X')

    dm_train = DataManager(image_shape[-1], (X_, Y_), path, bs, n_epochs, True, 'train')
    dm_test = DataManager(image_shape[-1], (X_, Y_), path, bs, 1, False, 'test')
    mon = nn.monitor.Monitor(model_name='CycleGAN', use_visdom=True, server='http://165.132.112.105')
    batches = dm_train.get_batches()
    fakes = [[], []]
    epoch = 0
    print('Training...')
    start = time.time()
    for it in batches:
        if it % dm_train.data_size:
            epoch += 1
            if epoch > (n_epochs >> 1):
                nn.anneal_learning_rate(lr_, epoch - (n_epochs >> 1), 'linear', num_iters=n_epochs-(n_epochs >> 1))

        if it == 0:
            x_, y_ = generate_X(), generate_Y()
            dis_x_loss_, dis_y_loss_ = train_dis(x_, y_)
        else:
            num = randint(0, len(fakes[0])-1)
            dis_x_loss_, dis_y_loss_ = train_dis(fakes[0][num], fakes[1][num])
        if np.isnan(dis_x_loss_ + dis_y_loss_) or np.isinf(dis_x_loss_ + dis_y_loss_):
            raise ValueError('Training failed! Stopped.')
        mon.plot('discriminator X loss', dis_x_loss_)
        mon.plot('discriminator Y loss', dis_y_loss_)

        x_fake_, y_fake_, gen_X_loss_, gen_Y_loss_, cycle_loss_ = train_gen()
        if np.isnan(gen_X_loss_ + gen_Y_loss_ + cycle_loss_) or np.isinf(gen_X_loss_ + gen_Y_loss_ + cycle_loss_):
            raise ValueError('Training failed! Stopped.')
        mon.plot('generator X from Y loss', gen_X_loss_)
        mon.plot('generator Y from X loss', gen_Y_loss_)
        mon.plot('cycle loss', cycle_loss_)

        if it < 50:
            fakes[0].append(x_fake_)
            fakes[1].append(y_fake_)
        else:
            num = randint(0, 49)
            fakes[0][num], fakes[1][num] = x_fake_, y_fake_

        if it % 1000 == 0:
            batches_test = dm_test.get_batches()
            X_reals, Y_reals, X_fakes, Y_fakes = [], [], [], []
            for _ in batches_test:
                if it == 0:
                    X_reals.append(dm_train.unnormalize(X_.get_value()))
                    Y_reals.append(dm_train.unnormalize(Y_.get_value()))
                X_fakes.append(generate_X() / 2. + .5)
                Y_fakes.append(generate_Y() / 2. + .5)
            
            if it == 0:
                mon.imwrite('X_real', np.concatenate(X_reals[:50]))
                mon.imwrite('Y_real', np.concatenate(Y_reals[:50]))

            mon.imwrite('X_fake', np.concatenate(X_fakes[:50]))
            mon.imwrite('Y_fake', np.concatenate(Y_fakes[:50]))
            mon.plot('elapsed time', (time.time() - start) / 60.)
            mon.flush()
        mon.tick()

    batches_test = dm_test.get_batches()
    for it in batches_test:
        mon.imwrite('X_fake_final_%d' % it, dm_train.unnormalize(generate_X()))
        mon.imwrite('Y_fake_final_%d' % it, dm_train.unnormalize(generate_Y()))
    mon.flush(use_visdom_for_image=False)
    print('Training finished!')


if '__main__' in __name__:
    train((3, 128, 128), 'C:/Users/justanhduc/Downloads/pytorch-CycleGAN-and-pix2pix-master/datasets/ukiyoe2photo')
