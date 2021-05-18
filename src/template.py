def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 7 #10
        args.n_resblocks = 7 #20
        args.n_feats = 16 #64
        args.chop = True
        args.reduction = 4

    if args.template.find('SplitSR') >= 0:
        args.model = 'SplitSR'
        args.n_resgroups = 7
        args.n_resblocks = 7
        args.n_feats = 16 #64
        args.alpha_ratio = 0.25
        args.hybrid_index = 3 #7

    if args.template.find('MySR') >= 0:
        args.model = 'MySR'
        args.n_resgroups = 6
        args.n_resblocks = 4
        args.n_feats = 48 #64
        args.alpha_ratio = 0.25
        args.hybrid_index = 4

    if args.template.find('TeachSR') >= 0:
        args.model = 'TeachSR'
        args.n_resgroups = 6
        args.n_resblocks = 12
        args.n_feats = 64 #64

    if args.template.find('MobileSR') >= 0:
        args.model = 'MobileSR'
        args.n_resgroups = 7
        args.n_resblocks = 7
        args.expand_ratio = 6
        args.n_feats = 16

    if args.template.find('ShuffleNet') >= 0:
        args.model = 'ShuffleNet'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 16
        args.alpha_ratio = 0.5

    if args.template.find('LatticeNet') >= 0:
        args.model = 'LatticeNet'
        args.n_feats = 16
        args.num_LBs = 4

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

