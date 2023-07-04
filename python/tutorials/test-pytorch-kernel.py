import torch 

if __name__ == '__main__':
    data_type = torch.float16
    iter_nums = 1

    test_dict_train = {
        'X x QKV w': (1, 8192, 4608, 12288),
        'Q x K^T': (192, 512, 512, 128),
        'QK^T x V': (192, 512, 128, 512),
        'Proj': (1, 8192, 12288, 1536),
        'FC1': (1, 8192, 6144, 12288),
        'FC2': (1, 8192, 12288, 6144),
        'Q x K^T Flat': (1, 512, 512, 24576),
        'QK^T x V Flat': (1, 512, 24576, 512)
    }

    test_dict_inference_w_o_KV = {
        'X x QKV w': (1, 8704, 4608, 12288),
        'Q x K^T': (192, 543, 543, 128),
        'QK^T x V': (192, 543, 128, 543),
        'Proj': (1, 8688, 12288, 1536),
        'FC1': (1, 8688, 6144, 12288),
        'FC2': (1, 8688, 12288, 6144),
        'Q x K^T Flat': (1, 543, 543, 24576),
        'QK^T x V Flat': (1, 543, 24576, 543)
    }

    test_dict_inference_w_KV = {
        'X x QKV w': (1, 16, 4608, 12288),
        'Q x K^T': (192, 1, 543, 128),
        'QK^T x V': (192, 1, 128, 543),
        'Proj': (1, 16, 12288, 1536),
        'FC1': (1, 16, 6144, 12288),
        'FC2': (1, 16, 12288, 6144),
        'Q x K^T Flat': (1, 1, 543, 24576),
        'QK^T x V Flat': (1, 1, 24576, 543)
    }

    # warm up
    x = torch.randn((114, 514), dtype=data_type).cuda()
    y = torch.randn((514, 114), dtype=data_type).cuda()
    z = torch.matmul(x, y)

    for test_dict in [test_dict_train, test_dict_inference_w_o_KV, test_dict_inference_w_KV]:
        print('==============================')
        for key, val in test_dict.items():
            B, M, N, K = val
            # test the time of torch.bmm
            a = torch.randn((B, M, K), dtype=data_type).cuda()
            b = torch.randn((B, K, N), dtype=data_type).cuda()
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(iter_nums):
                c = torch.bmm(a, b)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            print(f'torch.bmm {data_type} {key} {val} time: {start.elapsed_time(end)/iter_nums} ms', flush=True)
            