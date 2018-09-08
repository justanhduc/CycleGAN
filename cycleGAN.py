import numpy as np
from scipy import misc
from random import sample, shuffle, randint
import time
import os
import argparse
parser = argparse.ArgumentParser(description='Train a Cycle GAN.')
parser.add_argument('path', type=str, help='Path to a dataset folder. '
                                           'The folder must contain trainA, trainB, testA and testB subfolders.')
parser.add_argument('--out_size', default=128, type=int, help='Image size to be output.')
parser.add_argument('--bs', type=int, default=1, help='Batchsize.')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--lambd', type=float, default=10., help='Lambda value for cycle loss.')
parser.add_argument('--grad_pen_loss', type=float, default=0., help='Coefficient of fake and real gradient penalties.')
parser.add_argument('--spec_norm', type=int, default=0, help='Whether to use spectral normaliztion.')
parser.add_argument('--use_sigmoid', type=int, default=False, help='Use sigmoid for output activation.')
parser.add_argument('--checkpoint', type=int, default=0, help='Epoch to continue from.')
parser.add_argument('--checkpoint_files', type=str, default=None, nargs='+', help='Model weights to load.')
parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import theano
from theano import tensor as T

import neuralnet as nn

image_shape = (3, args.out_size, args.out_size)
path = args.path
bs = args.bs
lambd = args.lambd
grad_pen_loss = args.grad_pen_loss
lr = 2e-4
beta1 = .5
n_epochs = args.n_epochs
checkpoint = args.checkpoint
checkpoint_files = args.checkpoint_files


def resblock(input_shape, num_filters, block_name, **kwargs):
    block = nn.Sequential(input_shape=input_shape, layer_name=block_name)
    block.append(nn.ReflectPaddingConv(input_shape, num_filters, layer_name=block_name + '/conv1'))
    block.append(nn.ReflectPaddingConv(input_shape, num_filters, activation='linear', layer_name=block_name + '/conv2'))
    return block


class ResNetGenerator(nn.Sequential):
    def __init__(self, input_shape, num_filters=64, name='ResNetGen'):
        super(ResNetGenerator, self).__init__(input_shape=input_shape, layer_name=name)

        self.append(nn.ReflectPaddingConv(input_shape, num_filters, 7, layer_name=name + '/first'))
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
            nn.ReflectPaddingConv(self.output_shape, 3, 7, activation='tanh', use_batchnorm=False, layer_name=name + '/output'))


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
    def __init__(self, output_size, placeholders, path, batchsize, n_epochs, shuffle, type, checkpoint=0):
        super(DataManager, self).__init__(None, placeholders, path=path, batch_size=batchsize, n_epochs=n_epochs,
                                          shuffle=shuffle, checkpoint=checkpoint)
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


def unnormalize(input):
    return input / 2. + .5


def grad_pen(output, input):
    d_input = T.grad(T.sum(output), input)
    return T.sum(d_input ** 2)


def train():
    gen_x2y = ResNetGenerator((None,) + image_shape, name='Generator_X2Y')
    gen_y2x = ResNetGenerator((None,) + image_shape, name='Generator_Y2X')
    dis_x = Discriminator((None,) + image_shape, 64, name='DiscriminatorX', use_sigmoid=args.use_sigmoid)
    dis_y = Discriminator((None,) + image_shape, 64, name='DiscriminatorY', use_sigmoid=args.use_sigmoid)

    X = T.tensor4('inputX')
    Y = T.tensor4('inputY')
    X_fake_t = T.tensor4('inputX_fake')
    Y_fake_t = T.tensor4('inputY_fake')
    X_ = theano.shared(np.zeros((bs,) + image_shape, 'float32'), 'inputX placeholder')
    Y_ = theano.shared(np.zeros((bs,) + image_shape, 'float32'), 'inputY placeholder')
    lr_ = theano.shared(np.float32(args.lr), 'learning rate')

    nn.set_training_status(True)
    Y_fake = gen_x2y(X)
    X_fake = gen_y2x(Y)

    pred_X_real = dis_x(X)
    pred_X_fake = dis_x(X_fake_t)

    pred_Y_real = dis_y(Y)
    pred_Y_fake = dis_y(Y_fake_t)

    X_from_Y_fake = gen_y2x(Y_fake)
    Y_from_X_fake = gen_x2y(X_fake)

    loss_fn = nn.binary_cross_entropy if args.use_sigmoid else lambda x, y: T.mean((x - y) ** 2.)
    dis_X_loss = loss_fn(pred_X_real, T.ones_like(pred_X_real)) + loss_fn(pred_X_fake, T.zeros_like(pred_X_fake)) + \
                 grad_pen_loss * grad_pen(pred_X_real, X) + grad_pen_loss * grad_pen(pred_X_fake, X_fake_t)
    dis_Y_loss = loss_fn(pred_Y_real, T.ones_like(pred_Y_real)) + loss_fn(pred_Y_fake, T.zeros_like(pred_Y_fake)) + \
                 grad_pen_loss * grad_pen(pred_Y_real, Y) + grad_pen_loss * grad_pen(pred_Y_fake, Y_fake_t)
    gen_X_loss = loss_fn(dis_x(X_fake), T.ones_like(pred_Y_fake))
    gen_Y_loss = loss_fn(dis_y(Y_fake), T.ones_like(pred_X_fake))
    cycle_loss = nn.norm_error(X_from_Y_fake, X, 1) + nn.norm_error(Y_from_X_fake, Y, 1)

    loss_gen = gen_X_loss + gen_Y_loss + cycle_loss * lambd
    loss_dis = dis_X_loss + dis_Y_loss
    updates_gen = nn.adam(loss_gen, gen_x2y.trainable+gen_y2x.trainable, lr_, beta1)
    updates_dis = nn.adam(loss_dis, dis_x.trainable+dis_y.trainable, lr_, beta1)
    if args.spec_norm:
        updates_dis = nn.utils.spectral_normalize(updates_dis)
    train_gen = nn.function([], [X_fake, Y_fake, gen_X_loss, gen_Y_loss, cycle_loss], updates=updates_gen,
                            givens={X: X_, Y: Y_}, name='train generators')
    train_dis = nn.function([X_fake_t, Y_fake_t], [dis_X_loss, dis_Y_loss], updates=updates_dis, givens={X: X_, Y: Y_},
                            name='train discriminators')

    nn.set_training_status(False)
    generate_X = nn.function([], X_fake, givens={Y: Y_}, name='generate X from Y')
    generate_Y = nn.function([], Y_fake, givens={X: X_}, name='generate Y from X')

    dm_train = DataManager(image_shape[-1], (X_, Y_), path, bs, n_epochs, True, 'train', checkpoint=checkpoint)
    dm_test = DataManager(image_shape[-1], (X_, Y_), path, bs, 1, False, 'test')
    mon = nn.monitor.Monitor(model_name='CycleGAN', use_visdom=False, checkpoint=checkpoint*dm_train.data_size)
    if checkpoint_files:
        nn.utils.load_batch_checkpoints(checkpoint_files, gen_x2y.params + gen_y2x.params + dis_x.params + dis_y.params)

    batches = dm_train.get_batches()
    fakes = [[], []]
    epoch = checkpoint
    checkpoint_it = checkpoint * dm_train.data_size
    print('Training...')
    start = time.time()
    for it in batches:
        if it % dm_train.data_size == 0:
            epoch += 1
            if epoch > (n_epochs >> 1):
                nn.anneal_learning_rate(lr_, epoch - (n_epochs >> 1), 'linear', num_iters=n_epochs-(n_epochs >> 1))

        if it == checkpoint_it:
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

        if it < checkpoint_it + 50:
            fakes[0].append(x_fake_)
            fakes[1].append(y_fake_)
        else:
            num = randint(0, 49)
            fakes[0][num], fakes[1][num] = x_fake_, y_fake_

        if it % 1000 == 0 or it == checkpoint_it:
            batches_test = dm_test.get_batches()
            X_reals, Y_reals, X_fakes, Y_fakes = [], [], [], []
            for _ in batches_test:
                if it == checkpoint_it:
                    X_reals.append(unnormalize(X_.get_value()))
                    Y_reals.append(unnormalize(Y_.get_value()))
                X_fakes.append(generate_X() / 2. + .5)
                Y_fakes.append(generate_Y() / 2. + .5)

            mon.dump(nn.utils.shared2numpy(gen_x2y.params), 'GenX2Y.npz', keep=2)
            mon.dump(nn.utils.shared2numpy(gen_y2x.params), 'GenY2X.npz', keep=2)
            mon.dump(nn.utils.shared2numpy(dis_x.params), 'DisX.npz', keep=2)
            mon.dump(nn.utils.shared2numpy(dis_y.params), 'DisY.npz', keep=2)

            if it == checkpoint_it:
                mon.imwrite('X_real', np.concatenate(X_reals[:50]))
                mon.imwrite('Y_real', np.concatenate(Y_reals[:50]))

            mon.imwrite('X_fake', np.concatenate(X_fakes[:50]))
            mon.imwrite('Y_fake', np.concatenate(Y_fakes[:50]))
            mon.plot('elapsed time', (time.time() - start) / 60.)
            mon.flush()
        mon.tick()

    batches_test = dm_test.get_batches()
    for it in batches_test:
        mon.imwrite('X_fake_final_%d' % it, unnormalize(generate_X()))
        mon.imwrite('Y_fake_final_%d' % it, unnormalize(generate_Y()))
    mon.flush(use_visdom_for_image=False)

    mon.dump(nn.utils.shared2numpy(gen_x2y.params), 'GenX2Y-fin.npz')
    mon.dump(nn.utils.shared2numpy(gen_y2x.params), 'GenY2X-fin.npz')
    mon.dump(nn.utils.shared2numpy(dis_x.params), 'DisX-fin.npz')
    mon.dump(nn.utils.shared2numpy(dis_y.params), 'DisY-fin.npz')

    print('Training finished!')


if '__main__' in __name__:
    train()
