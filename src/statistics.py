from ptflops import get_model_complexity_info
import torch
from importlib import import_module
from option import args


def main():
    with torch.no_grad():
        # device = torch.device('cpu' if args.cpu else 'cuda')
        device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')
        module = import_module('model.' + args.model.lower())
        my_model = module.make_model(args).to(device)

        # input = torch.randn(1, 3, 678, 1020).to(device)
        flops, params = get_model_complexity_info(my_model, (3, 678, 1020), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        print("params =", params, ", flops =", flops)


if __name__ == '__main__':
    main()
