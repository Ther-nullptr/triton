python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 128 --block-n 256 --block-k 64 --group-m 8 --num-stages 3 --num-warps 8 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 64 --block-n 256 --block-k 32 --group-m 8 --num-stages 4 --num-warps 4 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 128 --block-n 128 --block-k 32 --group-m 8 --num-stages 4 --num-warps 4 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 128 --block-n 64 --block-k 32 --group-m 8 --num-stages 4 --num-warps 4 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 64 --block-n 128 --block-k 32 --group-m 8 --num-stages 4 --num-warps 4 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 128 --block-n 32 --block-k 32 --group-m 8 --num-stages 4 --num-warps 4 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 64 --block-n 32 --block-k 32 --group-m 8 --num-stages 5 --num-warps 2 
python 03-2-batched-matrix-multiplication-ncu-benchmark.py --batch 16 --block-m 32 --block-n 64 --block-k 32 --group-m 8 --num-stages 5 --num-warps 2 