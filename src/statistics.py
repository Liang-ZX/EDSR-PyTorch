from ptflops import get_model_complexity_info
import torch
from importlib import import_module
from option import args
from utils_modelsummary import get_model_activation, get_model_flops
import os


def main():
    with torch.no_grad():
        # device = torch.device('cpu' if args.cpu else 'cuda')
        device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

        module = import_module('model.' + args.model.lower())
        my_model = module.make_model(args).to(device)
        # if args.precision == 'half':
        #     my_model.half()
        # load(
        #     my_model,
        #     os.path.join(os.path.join('..', 'experiment', args.load), 'model'),
        #     pre_train=args.pre_train,
        #     resume=args.resume,
        #     cpu=args.cpu
        # )

        # input = torch.randn(1, 3, 678, 1020).to(device)
        flops, params = get_model_complexity_info(my_model, (3, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=False)
        max_memory = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
        activations, num_conv2d = get_model_activation(my_model, (3, 256, 256))

        # print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
        # print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

        # flops = get_model_flops(my_model, (3, 678, 1020), False)
        # print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))
        #
        # num_parameters = sum(map(lambda x: x.numel(), my_model.parameters()))
        # print('{:>16s} : {:<.4f} [k]'.format('#Params', num_parameters / 10 ** 3))
        print("#Params =", params, ", FLOPS =", flops)
        print('#Activations = {:<.4f} [M] , #Conv2d = {:<d}'.format(activations / 10 ** 6, num_conv2d))

        print('Max Memery = {:<.3f} [M]'.format(max_memory))


def load(load_model, apath, pre_train='', resume=-1, cpu=False):
    load_from = None
    kwargs = {}
    if cpu:
        kwargs = {'map_location': lambda storage, loc: storage}

    if resume == -1:
        load_from = torch.load(
            os.path.join(apath, 'model_latest.pt'),
            **kwargs
        )
    elif resume == 0:
        if pre_train == 'download':
            print('Download the model')
            dir_model = os.path.join('..', 'models')
            os.makedirs(dir_model, exist_ok=True)
            load_from = torch.utils.model_zoo.load_url(
                load_model.url,
                model_dir=dir_model,
                **kwargs
            )
        elif pre_train:
            print('Load the model from {}'.format(pre_train))
            load_from = torch.load(pre_train, **kwargs)
    else:
        load_from = torch.load(
            os.path.join(apath, 'model_{}.pt'.format(resume)),
            **kwargs
        )

    if load_from:
        load_model.load_state_dict(load_from, strict=False)


if __name__ == '__main__':
    main()
