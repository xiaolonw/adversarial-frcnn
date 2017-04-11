# ---------------------------------------------------------
# Copyright (c) 2015, Saurabh Gupta
# 
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------


import caffe, yaml 

def net_surgery(net, json_file_or_dict):
    # Load the JSON file
    if isinstance(json_file_or_dict, str):
        with open(json_file_or_dict, 'rt') as f:
            source_description = yaml.load(f)
    else:
        source_description = json_file_or_dict
    # Find a list of blobs in the target net
    target_blobs = net.params.keys() 
    target_blobs = dict(zip(target_blobs, [0]*len(target_blobs)))

    # For each item in the json file load the network and copy the layers
    for src_desc in source_description:
        net_source = caffe.Net(src_desc['prototxt'], src_desc['model'], caffe.TEST)
        for j in xrange(len(src_desc['copy_ops']['dest'])):
            dest_name = src_desc['copy_ops']['dest'][j]
            
            assert dest_name in target_blobs, \
                'Destination name {} not in target network blobs'.format(dest_name)
            
            src_name = src_desc['copy_ops']['source'][j]
            assert src_name in net_source.params.keys(), \
                'Source name {} not in source network blobs'.format(src_name)

            allow_different_shape = src_desc['copy_ops']['reshape'][j]
            
            if target_blobs[dest_name] is not 0:
                print 'Target blob {} is being reassigned'.format(dest_name)
            target_blobs[dest_name] = target_blobs[dest_name] + 1

            assert(len(net.params[dest_name]) == \
              len(net_source.params[src_name])), \
              'Number of blobs in {} in source do not match number of blobs in {} in destination'\
              .format(src_name, dest_name)

            for k in xrange(len(net.params[dest_name])):
                src = net_source.params[src_name][k]
                dest = net.params[dest_name][k]
                if allow_different_shape:
                    assert(src.count == dest.count), \
                      'Count of blobs in {}[{:d}] in source do not match count of blobs in {}[{:d}] in destination'\
                      .format(src_name, k, dest_name, k)
                    dest.data[...] = src.data.reshape(dest.data.shape)
                else:
                    src_shape = src.data.shape
                    dest_shape = dest.data.shape
                    assert(src_shape == dest_shape), \
                      'Shape of blobs in {}[{:d}] {} in source do not match shape of blobs in {}[{:d}] {} in destination'\
                      .format(src_name, k, str(src_shape), dest_name, k, str(dest_shape))
                    dest.data[...] = src.data

    unusual = [x for x in target_blobs.keys() if target_blobs[x] is not 1]
    for x in unusual:
        print 'Parameter blob {} copied {:d} times.'.format(x, target_blobs[x])

    return target_blobs



