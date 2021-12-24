import os
from options import parse_args

def run_model(room_num, method):
    args = parse_args()
    if method == "planercnn":
        from evaluate import evaluate
        args.methods = 'f'
        args.suffix = 'warping_refine'
    if method == "mws":
        from evaluate_planenet import evaluate
        args.methods = 't'
        args.suffix = 'gt'
    if method == "planenet":
        from evaluate_planenet import evaluate
        args.methods = 'p'
        args.suffix = 'warping_refine'
    args.dataset = 'inference'
    # args.dataset = 'my_dataset'
    args.customDataFolder = 'smartphone_indoor/%d_%s' % (room_num, method)
    args.test_dir = 'inference/%d_%s' % (room_num, method)
    try:
        os.system("rm -r " + args.test_dir)
        os.system("mkdir -p %s" % args.test_dir)
    except OSError:
        print('Error: Creating directory. ' + args.test_dir)
    evaluate(args)