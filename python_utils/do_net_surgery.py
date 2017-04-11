# ---------------------------------------------------------
# Copyright (c) 2015, Saurabh Gupta
# 
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------

# For fusing network outputs
import _init_paths
import caffe
import pycaffe_utils
import sys, pprint, argparse

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Network surgery script')
  parser.add_argument('--out_net_def', help='prototxt file defining the output network', default=None, type=str)
  parser.add_argument('--net_surgery_json', help='json file which defines what blobs to copy from where', default=None, type=str)
  parser.add_argument('--out_net_file', help='caffemodel to save the ouput network to', default=None, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  net = caffe.Net(args.out_net_def, caffe.TEST)
  pycaffe_utils.net_surgery(net, args.net_surgery_json)
  net.save(args.out_net_file)
