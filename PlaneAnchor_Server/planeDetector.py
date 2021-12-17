import os
from evaluate import evaluate
from options import parse_args

def run_model(room_num, method):
    args = parse_args()
    args.methods = 'f'
    args.suffix = 'warping_refine'
    args.dataset = 'inference'
    args.customDataFolder = 'smartphone_indoor/%d_%s' % (room_num, method)
    args.test_dir = 'inference/%d_%s' % (room_num, method)
    try:
        os.system("rm -r " + args.test_dir)
        os.system("mkdir -p %s" % args.test_dir)
    except OSError:
        print('Error: Creating directory. ' + args.test_dir)
    evaluate(args)